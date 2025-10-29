from typing import List
import json
from pathlib import Path


def get_similar_captions(name, ids: List[int], lang: str):
    input_path = Path("outputs") / f"{name}_data.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Can't get similar captions because input_path not found: {input_path}")

    captions_list = []
    with open(input_path, encoding = "UTF-8") as f:
        data = json.load(f)    

    for id in ids:
        caption = data[id][lang]
        captions_list.append(caption)
    return captions_list    

