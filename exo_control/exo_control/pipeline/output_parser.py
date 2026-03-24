import json
import re
from typing import Dict, Any, Optional


ALLOWED_WEIGHT = {"LOW", "MEDIUM", "HEAVY"}
ALLOWED_FRAGILITY = {"LOW", "MEDIUM", "HIGH"}


def extract_json_block(text: str) -> Optional[str]:
    """
    Extract JSON block from model output.
    Handles cases like:
    - plain JSON
    - ```json ... ```
    - extra text around JSON
    """

    if not text:
        return None

    # fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    # plain {...}
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        candidate = brace.group(1)
        last = candidate.rfind("}")
        return candidate[: last + 1]

    return None


def validate_levels(weight: str, fragility: str) -> bool:
    weight = weight.upper()
    fragility = fragility.upper()

    return (weight in ALLOWED_WEIGHT) and (fragility in ALLOWED_FRAGILITY)


def parse_qwen_output(raw_text: str) -> Dict[str, Any]:
    """
    Parse Qwen2.5-VL output JSON.

    Returns dict with:
        json_ok
        object_name
        weight_level
        fragility_level
        raw_text
    """

    result = {
        "json_ok": False,
        "object_name": "",
        "weight_level": "",
        "fragility_level": "",
        "raw_text": raw_text,
    }

    json_str = extract_json_block(raw_text)

    if json_str is None:
        return result

    try:
        data = json.loads(json_str)
    except Exception:
        return result

    object_name = str(data.get("object_name", "")).strip()
    weight_level = str(data.get("weight_level", "")).strip().upper()
    fragility_level = str(data.get("fragility_level", "")).strip().upper()

    if not validate_levels(weight_level, fragility_level):
        return result

    result["json_ok"] = True
    result["object_name"] = object_name
    result["weight_level"] = weight_level
    result["fragility_level"] = fragility_level

    return result
