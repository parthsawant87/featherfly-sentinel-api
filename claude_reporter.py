# claude_reporter.py — SENTINEL Claude API Reporter
# Generates: inspection reports, Q&A answers, batch summaries
# Model: claude-haiku-4-5-20251001 (fast + cheap)
# Set env: ANTHROPIC_API_KEY=your_key

import anthropic, os
from rca_engine import RCAResult

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL  = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are SENTINEL, an expert semiconductor process engineer AI assistant.
You help QC engineers understand wafer and PCB defect classification results.

Rules:
- Be technically precise — use correct fab terminology
- Keep reports under 150 words unless asked for detail
- Always state: defect class, severity, most likely root cause, recommended action
- If confidence < 60%, flag it prominently
- Never guess beyond what the data supports
- Format using short paragraphs, not bullet lists
"""

def generate_inspection_report(rca: RCAResult) -> str:
    """Generate a concise inspection report for a single classification result."""
    conf_flag = " [LOW CONFIDENCE — VERIFY MANUALLY]" if rca.low_confidence else ""
    prompt = f"""Generate a concise inspection report for this SENTINEL result:

Defect Class: {rca.defect_class}{conf_flag}
Confidence: {rca.confidence*100:.1f}%
Severity: {rca.severity}
Primary Root Cause: {rca.primary_cause}
Recommended Action: {rca.action}
Fab Impact: {rca.fab_impact}
Process Checks: {', '.join(rca.process_check[:2])}

Write a 3-sentence report: (1) what was detected and severity, (2) most likely root cause, (3) recommended immediate action."""

    resp = client.messages.create(
        model=MODEL, max_tokens=220,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user", "content":prompt}]
    )
    return resp.content[0].text

def answer_engineer_question(question: str, context: dict, history: list=None) -> str:
    """Answer a QC engineer's question about the current inspection result.
    context: dict with defect_class, confidence, rca fields
    history: list of {role, content} for multi-turn conversation"""
    msgs = (history or [])[-6:] + [{
        "role": "user",
        "content": f"Inspection context: {context}\n\nQuestion: {question}"
    }]
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        system=SYSTEM_PROMPT, messages=msgs
    )
    return resp.content[0].text

def generate_batch_summary(wafer_map_data: dict) -> str:
    """Summarize an entire wafer scan in plain English."""
    prompt = f"""Summarize this wafer inspection scan for a process engineer:

Scan ID: {wafer_map_data['scan_id']}
Total cells inspected: {wafer_map_data['total_cells']}
Wafer yield: {wafer_map_data['yield_pct']:.1f}%
Defect distribution: {wafer_map_data['summary']}

Write a 4-sentence executive summary: (1) overall yield and pass/fail, (2) dominant defect type and severity, (3) most likely process root cause, (4) recommended immediate action for the process team."""

    resp = client.messages.create(
        model=MODEL, max_tokens=280,
        system=SYSTEM_PROMPT,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.content[0].text
