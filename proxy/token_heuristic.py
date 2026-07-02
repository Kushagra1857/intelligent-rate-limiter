import re
CALIBRATION_MULTIPLIER: float = 1.0
_SPLIT_PATTERN = re.compile('[\\s\\.,;:!?(){}\\[\\]\\"\'`<>|/\\\\@#$%^&*+=~\\-]+')

def estimate_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    pieces = _SPLIT_PATTERN.split(text)
    raw_count = sum((1 for p in pieces if p))
    return max(0, round(raw_count * CALIBRATION_MULTIPLIER))
