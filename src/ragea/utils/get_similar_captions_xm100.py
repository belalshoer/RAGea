from datasets import load_dataset
from vectorstores import VectorStore
from pathlib import Path
from typing import List
import tqdm
import json
import os

def preprocess_similar_captions_xm100(vector_store: VectorStore, k: int = 4, search_type: str = "similarity"):
    print("Loading xm100..")
    ds = load_dataset("neulab/PangeaBench-xm100", split = "en")
    print("Done Loading!")
    images = [(row["image"], row["image_id"])  for row in ds]

    similar_captions = {}

    print(f"Preprocessing top-{k} captions for all images in XM100 using vectorstore: {vector_store.name}:")

    for image, img_id in tqdm.tqdm(images):
        retrieved_data = vector_store.retrieve(img = image, k = k, search_type=search_type) 
        similar_captions[img_id] = retrieved_data

    out_dir = Path("outputs") / "similar_captions" / f"{vector_store.name}" 
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"top_{k} captoins using {search_type}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(similar_captions, f, ensure_ascii=False, indent=2) 

def get_captions_by_language(name, source: str, ids: List[int], lang: str):
    input_path = Path("outputs") / "coco_captions" / f"{source}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Can't get captions by other languages because Path: {input_path} doesn't exist.")


    with open(input_path, encoding = "UTF-8") as f:
        data = json.load(f)    

    id2caption = {i: data[i][lang] for i in ids}
    return id2caption  

def get_similar_captions_xm100(vector_store_name: str, lang: str, k: int = 4, search_type: str = "similarity"):
    similar_captions_path = Path("outputs") / "similar_captions" / f"{vector_store_name}" / f"top_{k} captoins using {search_type}.json"

    with open(similar_captions_path, "r", encoding="utf-8") as f:
        similar_captions = json.load(f)

    images_data = [
        [(info["img_id"], info["source"]) for info in similar_captions[img]]
        for img in similar_captions.keys()
    ]

    images_data_by_src = {}
    for group in images_data:
        for img_id, src in group:
            if not images_data_by_src.get(src, None):
                images_data_by_src[src] = [img_id]
            else:
                images_data_by_src[src].append(img_id)

    id2caption = {}
    for src in images_data_by_src.keys():
        id2caption.update(
            get_captions_by_language(name=vector_store_name, source=src, ids=images_data_by_src[src], lang=lang)
        )

    similar_captions_lang = [
        [id2caption[i] for i, _ in group]
        for group in images_data
    ]

    return similar_captions_lang