from typing import List
import json

input_path = "/home/belal.shoer/Desktop/RAGea/src/ragea/outputs/captions_35L.json"
def get_similar_captions(ids: List[int], lang: str):
    captions_list = []
    with open(input_path, encoding = "UTF-8") as f:
        data = json.load(f)    

    for id in ids:
        caption = data[id][lang]
        captions_list.append(caption)
    return captions_list    

