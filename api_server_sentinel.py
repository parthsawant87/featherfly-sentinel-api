# api_server_sentinel.py — FIXED VERSION

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np, io, time, os, asyncio, json
from PIL import Image
from pathlib import Path

import config as cfg
from rca_engine import analyze
from claude_reporter import generate_inspection_report, answer_engineer_question
from db_logger import log_prediction, can_call_claude, record_claude_call, get_claude_usage_today
from active_learner import flag_if_uncertain
from spc_monitor import check_spc

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = FastAPI(title="Featherfly SENTINEL API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────
# GLOBALS (FIXED)
# ─────────────────────────────────────────────
_sse_subscribers = []
_latest_result = None

_interp = None
_IN = None
_OUT = None

# ─────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
_TFLITE_PATH = BASE_DIR / "models" / "sentinel_int8.tflite"

print("MODEL PATH:", _TFLITE_PATH)
print("FILE EXISTS:", _TFLITE_PATH.exists())

# ─────────────────────────────────────────────
# LOAD TFLITE LIB
# ─────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite_lib
except ImportError:
    import tensorflow.lite as tflite_lib

# ─────────────────────────────────────────────
# LAZY MODEL LOADING (FIXED)
# ─────────────────────────────────────────────
def get_interpreter():
    global _interp, _IN, _OUT

    if _interp is None:
        print("[model] Loading TFLite model...")

        _interp = tflite_lib.Interpreter(model_path=str(_TFLITE_PATH))
        _interp.allocate_tensors()

        _IN = _interp.get_input_details()[0]
        _OUT = _interp.get_output_details()[0]

        print("[model] ✓ Model loaded successfully")

    return _interp, _IN, _OUT


# ─────────────────────────────────────────────
# SSE
# ─────────────────────────────────────────────
async def _notify_sse(result: dict):
    dead = []
    for q in _sse_subscribers:
        try:
            await q.put(result)
        except Exception:
            dead.append(q)
    for q in dead:
        _sse_subscribers.remove(q)


# ─────────────────────────────────────────────
# PREPROCESS (FIXED)
# ─────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    _, IN, _ = get_interpreter()

    arr = np.array(img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), dtype=np.float32)
    arr = (arr / 255.0 - np.array(cfg.IMG_MEAN)) / np.array(cfg.IMG_STD)
    arr = np.transpose(arr, (2, 0, 1))
    arr = arr[np.newaxis]

    in_sc, in_zp = IN["quantization"]
    return (arr / in_sc + in_zp).astype(np.int8)


# ─────────────────────────────────────────────
# INFERENCE (FIXED)
# ─────────────────────────────────────────────
def get_interpreter():
    global _interp, _IN, _OUT

    print("STEP 1: enter get_interpreter")

    if _interp is None:
        print("STEP 2: creating interpreter")

        _interp = tflite_lib.Interpreter(
            model_path=str(_TFLITE_PATH),
            num_threads=1
        )

        print("STEP 3: interpreter created")

        _interp.allocate_tensors()

        print("STEP 4: tensors allocated")

        _IN = _interp.get_input_details()[0]
        _OUT = _interp.get_output_details()[0]

        print("STEP 5: model ready")

    return _interp, _IN, _OUT


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _latest_result

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")

        cls, conf, probs = run_inference(img)
        rca = analyze(cls, conf)

        if can_call_claude():
            report = generate_inspection_report(rca)
            record_claude_call()
        else:
            report = f"{rca.severity}: {rca.primary_cause}. Action: {rca.action}"

        result = {
            "predicted_class": cls,
            "confidence": round(conf, 4),
            "severity": rca.severity,
            "root_cause": rca.primary_cause,
            "action": rca.action,
            "report": report,
            "low_confidence": rca.low_confidence,
            "probabilities": {
                cfg.CLASS_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(probs)
            },
            "timestamp": time.time(),
            "filename": file.filename,
        }

        _latest_result = result

        log_prediction(result, module="sentinel")
        flag_if_uncertain(result)

        await _notify_sse(result)

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# SSE STREAM
# ─────────────────────────────────────────────
@app.get("/stream")
async def sse_stream():
    queue = asyncio.Queue()
    _sse_subscribers.append(queue)

    async def event_generator():
        yield ": heartbeat\n\n"

        if _latest_result:
            yield f"data: {json.dumps(_latest_result)}\n\n"

        try:
            while True:
                try:
                    result = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(result)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    context: dict = {}
    history: list = []


@app.post("/chat")
async def chat(req: ChatRequest):
    ctx = req.context or _latest_result
    answer = answer_engineer_question(req.question, ctx, req.history)
    return {"answer": answer}


# ─────────────────────────────────────────────
# OTHER ROUTES
# ─────────────────────────────────────────────
@app.get("/latest")
def latest():
    if not _latest_result:
        return {"status": "no_result_yet"}
    return _latest_result


@app.get("/spc")
def spc_status():
    return check_spc()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "sentinel-tflite-int8",
        "classes": cfg.NUM_CLASSES,
        "version": "1.0.0",
        "claude_budget": get_claude_usage_today(),
        "sse_clients": len(_sse_subscribers),
    }


# ─────────────────────────────────────────────
# LOCAL RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server_sentinel:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )