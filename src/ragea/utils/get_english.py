
import json

input_path = "/home/belal.shoer/Desktop/RAGea/COCO-35L/merged.jsonl"  
output_path = "/home/belal.shoer/Desktop/RAGea/COCO-35L/english_only.jsonl"

seen = set()
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        image_id = data["image_id"]
        if image_id not in seen and data["src_lang"] == "en":
            outfile.write(json.dumps({
                "image_id": image_id,
                "caption_tokenized": data["caption_tokenized"]
            }, ensure_ascii=False) + "\n")
            seen.add(image_id)

print(f"Done! Extracted {len(seen)} English captions to {output_path}")
