# rca_engine.py — SENTINEL Root Cause Analysis Engine
# Maps defect class → process root cause → action required
# Used by API and Claude reporter

from dataclasses import dataclass
from typing import List

RCA_KB = {
    "BRIDGE": {
        "severity":      "CRITICAL",
        "primary_cause": "Photolithography over-exposure or resist flow during post-exposure bake",
        "secondary":     ["Excessive exposure dose", "Resist reflow above Tg", "Mask defect", "Alignment error"],
        "process_check": ["Review exposure dose (mJ/cm²)", "Check PEB temperature uniformity",
                          "Inspect photomask for chrome defects", "Verify alignment mark quality"],
        "action":        "REWORK or SCRAP — short circuit risk. Check litho settings immediately.",
        "fab_impact":    "Electrical short — die fails continuity test",
    },
    "CRACK": {
        "severity":      "CRITICAL",
        "primary_cause": "Mechanical stress exceeding fracture toughness of substrate or film",
        "secondary":     ["Thermal shock during rapid heating/cooling", "Dicing saw vibration",
                          "Film stress mismatch", "Wafer edge handling"],
        "process_check": ["Review thermal ramp rates", "Check wafer chuck vacuum",
                          "Measure film stress (wafer bow)", "Inspect dicing blade condition"],
        "action":        "SCRAP — crack propagates under electrical/thermal stress in field.",
        "fab_impact":    "Structural failure — crack grows under thermal cycling in product",
    },
    "VIA": {
        "severity":      "CRITICAL",
        "primary_cause": "Via etch undercut, incomplete via fill, or mask misalignment",
        "secondary":     ["Etch chemistry imbalance", "Barrier layer deposition failure",
                          "CMP over-polish of via cap"],
        "process_check": ["Cross-section TEM of via", "Check etch selectivity to stop layer",
                          "Measure via resistance (Kelvin probe)"],
        "action":        "ELECTRICAL TEST — measure via chain resistance before proceeding.",
        "fab_impact":    "Open circuit or high resistance in interconnect stack",
    },
    "OPEN": {
        "severity":      "CRITICAL",
        "primary_cause": "Over-etch breaking conductor line or resist adhesion failure",
        "secondary":     ["Metal etch over-run", "Resist delamination", "Scratch during handling"],
        "process_check": ["Review metal etch endpoint detection", "Check resist adhesion (HMDS treatment)",
                          "Inspect wafer handling robots for contact damage"],
        "action":        "SCRAP or electrical test — open circuit, die will fail.",
        "fab_impact":    "Broken conductor — electrical open circuit",
    },
    "PARTICLE": {
        "severity":      "HIGH",
        "primary_cause": "Cleanroom contamination or process tool particle shedding",
        "secondary":     ["Cleanroom HEPA filter breach", "Etch chamber wall flaking",
                          "Human-introduced contamination", "Slurry agglomeration (CMP)"],
        "process_check": ["Run particle counter in cleanroom zones", "Clean/inspect etch chamber",
                          "Check gown protocol compliance", "Inspect slurry filters"],
        "action":        "CLEAN and re-inspect. Isolate contamination source. May rework if early stage.",
        "fab_impact":    "Size-dependent — large particles cause bridging or opens",
    },
    "CMP": {
        "severity":      "MEDIUM",
        "primary_cause": "CMP polish rate non-uniformity — pad wear or slurry flow imbalance",
        "secondary":     ["Pad glazing (polishing pad wear)", "Slurry flow rate variation",
                          "Wafer carrier pressure non-uniformity", "Pad conditioner wear"],
        "process_check": ["Measure post-CMP film thickness (spectroscopic ellipsometry)",
                          "Replace/condition polishing pad", "Verify slurry flow rate calibration"],
        "action":        "ADJUST CMP recipe. Monitor wafer bow/WIWNU. May recover with repolish.",
        "fab_impact":    "Within-wafer non-uniformity → yield gradient across wafer",
    },
    "LER": {
        "severity":      "MEDIUM",
        "primary_cause": "Photoresist sensitivity or exposure dose variation causing rough line edges",
        "secondary":     ["Resist molecular weight distribution", "Exposure dose uniformity",
                          "Developer concentration drift", "Standing wave effects"],
        "process_check": ["CD-SEM measurement of LER (3σ target <2nm for advanced nodes)",
                          "Dose-to-size calibration", "Developer replenishment rate"],
        "action":        "MONITOR — acceptable if LER < 10% of feature CD. Tune litho recipe.",
        "fab_impact":    "Transistor Vt variability → parametric yield loss at speed testing",
    },
    "OTHERS": {
        "severity":      "REVIEW",
        "primary_cause": "Unknown — does not match any known defect signature in model training set",
        "secondary":     ["Novel defect type", "Low confidence classification", "Image quality issue"],
        "process_check": ["Human review required", "SEM/EDX analysis recommended",
                          "Add to training set if new defect type confirmed"],
        "action":        "HOLD — route to engineer for manual review. Do not process further.",
        "fab_impact":    "Unknown — evaluate case by case",
    },
    "CLEAN": {
        "severity":      "PASS",
        "primary_cause": "No defect detected — region meets quality criteria",
        "secondary":     [],
        "process_check": [],
        "action":        "PASS — continue normal processing.",
        "fab_impact":    "None — good die",
    },
}

@dataclass
class RCAResult:
    defect_class:  str
    confidence:    float
    severity:      str
    primary_cause: str
    secondary:     List[str]
    process_check: List[str]
    action:        str
    fab_impact:    str
    low_confidence: bool

def analyze(defect_class: str, confidence: float) -> RCAResult:
    kb = RCA_KB.get(defect_class, RCA_KB["OTHERS"])
    return RCAResult(
        defect_class  = defect_class,
        confidence    = confidence,
        severity      = kb["severity"],
        primary_cause = kb["primary_cause"],
        secondary     = kb["secondary"],
        process_check = kb["process_check"],
        action        = kb["action"],
        fab_impact    = kb["fab_impact"],
        low_confidence= confidence < 0.60,
    )
