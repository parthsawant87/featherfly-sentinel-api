# api_server_sentinel.py — SENTINEL FastAPI Production Server
# Endpoints:
#   POST /predict       — classify single uploaded image
#   POST /wafer-map     — process batch of grid images
#   POST /chat          — Claude Q&A about inspection results
#   GET  /stream        — Server-Sent Events live push (~50ms latency)
#   GET  /latest        — most recent classification (SSE fallback)
#   GET  /spc           — SPC EWMA drift status
#   GET  /health        — liveness check
#
# Run on RPi:    gunicorn api_server_sentinel:app --workers 1 --bind 0.0.0.0:8000
# Run locally:   uvicorn api_server_sentinel:app --reload --port 8000
# Run on Render: gunicorn api_server_sentinel:app --workers 1 --bind 0.0.0.0:$PORT

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse   # ← StreamingResponse added
from pydantic import BaseModel
import numpy as np, io, time, os, asyncio, json                # ← asyncio, json added
from PIL import Image
import config as cfg
from rca_engine import analyze
from claude_reporter import generate_inspection_report, answer_engineer_question
from db_logger import log_prediction, can_call_claude, record_claude_call
from active_learner import flag_if_uncertain
from spc_monitor import check_spc

app = FastAPI(title="Featherfly SENTINEL API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Load TFLite model at startup ──────────────────────────────────────────
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_TFLITE_PATH = BASE_DIR / "models" / "sentinel_int8.tflite"

try:
    import tflite_runtime.interpreter as tflite_lib
except ImportError:
    import tensorflow.lite as tflite_lib

_interp = tflite_lib.Interpreter(model_path=_TFLITE_PATH)
_interp.allocate_tensors()
_IN  = _interp.get_input_details()[0]
_OUT = _interp.get_output_details()[0]

_latest_result: dict = {}
_sse_subscribers: list = []          # list of asyncio.Queue for connected SSE clients


# ── SSE: notify all connected dashboard clients ───────────────────────────
async def _notify_sse(result: dict):
    """Push result to every connected EventSource client immediately."""
    dead = []
    for q in _sse_subscribers:
        try:
            await q.put(result)
        except Exception:
            dead.append(q)
    for q in dead:
        _sse_subscribers.remove(q)


# ── Inference helpers ─────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    arr = np.array(img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE)), dtype=np.float32)
    arr = (arr / 255.0 - np.array(cfg.IMG_MEAN)) / np.array(cfg.IMG_STD)
    arr = np.transpose(arr, (2, 0, 1))        # HWC → CHW
    arr = arr[np.newaxis]                      # [1, 3, H, W]
    in_sc, in_zp = _IN["quantization"]
    return (arr / in_sc + in_zp).astype(np.int8)

def run_inference(img: Image.Image) -> tuple:
    _interp.set_tensor(_IN["index"], preprocess(img))
    _interp.invoke()
    out_sc, out_zp = _OUT["quantization"]
    q      = _interp.get_tensor(_OUT["index"])[0]
    logits = (q.astype(np.float32) - out_zp) * out_sc
    probs  = np.exp(logits) / np.exp(logits).sum()
    pi     = int(probs.argmax())
    return cfg.CLASS_NAMES[pi], float(probs[pi]), probs


# ── Routes ────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _latest_result
    try:
        data             = await file.read()
        img              = Image.open(io.BytesIO(data)).convert("RGB")
        cls, conf, probs = run_inference(img)
        rca              = analyze(cls, conf)

        # Claude report — only if daily budget not exhausted
        if can_call_claude():
            report = generate_inspection_report(rca)
            record_claude_call()
        else:
            report = f"{rca.severity}: {rca.primary_cause}. Action: {rca.action}"

        result = {
            "predicted_class": cls,
            "confidence":      round(conf, 4),
            "severity":        rca.severity,
            "root_cause":      rca.primary_cause,
            "action":          rca.action,
            "report":          report,
            "low_confidence":  rca.low_confidence,
            "probabilities":   {cfg.CLASS_NAMES[i]: round(float(p), 4)
                                 for i, p in enumerate(probs)},
            "timestamp":       time.time(),
            "filename":        file.filename,
        }
        _latest_result = result
        log_prediction(result, module="sentinel")
        flag_if_uncertain(result)              # queue low-confidence for review

        # Push to all SSE subscribers — instant dashboard update
        await _notify_sse(result)

        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/stream")
async def sse_stream():
    """Server-Sent Events endpoint.
    Dashboard connects once via EventSource('/stream').
    Result is pushed the instant /predict completes — ~50ms vs 3s polling.
    Heartbeat comment sent every 15s to keep connection alive through proxies.
    """
    queue = asyncio.Queue()
    _sse_subscribers.append(queue)

    async def event_generator():
        # Send heartbeat immediately so client knows connection is alive
        yield ": heartbeat\n\n"
        # Send last known result if available
        if _latest_result:
            yield f"data: {json.dumps(_latest_result)}\n\n"
        try:
            while True:
                try:
                    result = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(result)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"    # keepalive — not a data event
        except asyncio.CancelledError:
            # Client disconnected — clean up
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",           # disables Nginx/Render proxy buffering
        }
    )


class ChatRequest(BaseModel):
    question: str
    context:  dict = {}
    history:  list = []

@app.post("/chat")
async def chat(req: ChatRequest):
    ctx    = req.context or _latest_result
    answer = answer_engineer_question(req.question, ctx, req.history)
    return {"answer": answer}


@app.get("/latest")
def latest():
    """Fallback polling endpoint — used when SSE connection is down."""
    if not _latest_result:
        return JSONResponse({"status": "no_result_yet"})
    return JSONResponse(_latest_result)


@app.get("/spc")
def spc_status():
    """Returns current EWMA SPC status and control limits.
    Use this in the dashboard SPC widget.
    """
    return check_spc()


@app.get("/health")
def health():
    from db_logger import get_claude_usage_today
    return {
        "status":        "ok",
        "model":         "sentinel-tflite-int8",
        "classes":       cfg.NUM_CLASSES,
        "version":       "1.0.0",
        "claude_budget": get_claude_usage_today(),
        "sse_clients":   len(_sse_subscribers),
    }

#----fix error-------------------------------------------------
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api_server_sentinel:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
