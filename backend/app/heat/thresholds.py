"""WBGT risk-band classification.

ISO 7243 and ACGIH TLV(R) heat-stress action limits both vary by work
intensity (metabolic rate) AND by acclimatization state -- there is no single
universal WBGT threshold. work_intensity is therefore a required parameter,
not a hardcoded constant: callers must say what activity level they're
screening for (a resting pedestrian and a heavy-labor construction worker do
not share a danger threshold).

WBGT_THRESHOLDS_C below is a simplified 4-band cut (SAFE/CAUTION/DANGER/
CRITICAL) loosely modeled on ACGIH TLV action-limit shapes for ACCLIMATIZED
workers, intended for a general-public-facing heat-exposure app rather than
occupational-safety compliance. It is NOT a substitute for the full ACGIH TLV
tables (which also vary by work/rest duty cycle) -- consult those directly
for any occupational-safety decision.
"""

from __future__ import annotations

# Upper bound (inclusive) of each band, in degrees C WBGT, per work intensity.
# Thresholds fall as work intensity rises: harder work generates more internal
# heat, so a given ambient WBGT is riskier at heavier exertion.
WBGT_THRESHOLDS_C: dict[str, dict[str, float]] = {
    "rest": {"SAFE": 33.0, "CAUTION": 34.5, "DANGER": 36.0},  # ~0 W metabolic rate
    "light": {"SAFE": 30.0, "CAUTION": 31.5, "DANGER": 33.0},  # ~180-300 W (walking, light tasks)
    "moderate": {"SAFE": 28.0, "CAUTION": 29.5, "DANGER": 31.0},  # ~300-415 W (brisk walking, moderate labor)
    "heavy": {"SAFE": 25.5, "CAUTION": 27.0, "DANGER": 28.5},  # ~415-520 W (heavy labor)
}
# Anything above the DANGER cut point for the given intensity is CRITICAL.

BANDS_IN_ORDER = ("SAFE", "CAUTION", "DANGER", "CRITICAL")


def classify(wbgt_c: float, work_intensity: str = "moderate") -> str:
    """Classify a WBGT value (C) into SAFE/CAUTION/DANGER/CRITICAL for the given work intensity."""
    if work_intensity not in WBGT_THRESHOLDS_C:
        raise ValueError(f"Unknown work_intensity {work_intensity!r}; choose one of {sorted(WBGT_THRESHOLDS_C)}")

    cuts = WBGT_THRESHOLDS_C[work_intensity]
    if wbgt_c <= cuts["SAFE"]:
        return "SAFE"
    if wbgt_c <= cuts["CAUTION"]:
        return "CAUTION"
    if wbgt_c <= cuts["DANGER"]:
        return "DANGER"
    return "CRITICAL"
