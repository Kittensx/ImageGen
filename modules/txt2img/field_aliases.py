
def parse_size_field(value: str) -> dict[str, int]:
    text = str(value).strip().lower()
    if "x" not in text:
        return {}

    left, right = text.split("x", 1)

    try:
        width = int(left.strip())
        height = int(right.strip())
    except ValueError:
        return {}

    return {
        "width": width,
        "height": height,
    }
    
FIELD_ALIASES = {
  "prompt": "prompt",
  "negative prompt": "negative_prompt",
  "steps": "steps",
  "cfg scale": "cfg_scale",
  "seed": "seed",
  "sampler": "sampler_name",
  "scheduler": "scheduler_name",
  "size": "size",
  "model": "model_path",
  "model hash": "model_hash",
}

SPECIAL_FIELD_HANDLERS = {
  "size": parse_size_field,
}