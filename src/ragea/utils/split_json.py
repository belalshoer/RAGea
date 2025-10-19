import json, math
from typing import List  

def split_json(grouped_path: str, output_prefix: str, parts: int = 4) -> List[str]:
    with open(grouped_path, encoding="utf-8") as f:
        data = json.load(f)

    keys = list(data.keys())              
    n = len(keys)
    step = math.ceil(n / parts)           
    shard_paths = []

    for i in range(parts):                 
        chunk_keys = keys[i * step : (i + 1) * step]
        shard_data = {k: data[k] for k in chunk_keys}
        output_path = f"{output_prefix}_part{i+1}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(shard_data, f, ensure_ascii=False, indent=2)
        shard_paths.append(output_path)


    return shard_paths
