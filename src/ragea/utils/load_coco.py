import json
import tqdm
from pathlib import Path

def load_coco(input_path: Path, output_path:Path):
    print(f"loading data in  {input_path}: ")

    items = {}
    
    with open(input_path, encoding = "UTF-8") as f:
        for line in tqdm.tqdm(f.readlines()):
            line = line.strip()
            if not line:
                continue
            
            caption = json.loads(line)
            img_id = caption["image_id"]

            if not items.get(img_id, None):
                items[img_id] = {}

            if not items[img_id].get("en", None):
                items[img_id]["en"] = caption["caption_tokenized"]

            items[img_id][caption["trg_lang"]] = caption["translation_tokenized"]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2) 
                    