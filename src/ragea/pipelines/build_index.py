from vectorstores import FaissVectorStore
from utils import load_coco, prepare_documents, split_json
import os, shutil

def build_add_index_sharded(name: str, input_path: str, *, parts: int = 4, overwrite: bool = False):
    grouped_path = f"outputs/{name}_data.json"
    output_prefix = f"outputs/{name}_data"

    load_coco(input_path=input_path, output_path=grouped_path)
    shard_paths = split_json(grouped_path, output_prefix=output_prefix, parts=parts)

    vs = FaissVectorStore(name)
    index_dir = os.path.join("vector_stores", name)
    if os.path.isdir(index_dir) and overwrite:
        print(f"[sharded] Overwrite=True → deleting {index_dir}")
        shutil.rmtree(index_dir)

    for shard_index, shard_grouped in enumerate(shard_paths, start=1):   
        docs = prepare_documents(shard_grouped, input_path)
        if shard_index == 1 and (overwrite or not os.path.isdir(index_dir)):
            print(f"[sharded] BUILD from shard {shard_index}/{parts}: {shard_grouped}")
            vs.create_from_documents(docs)
        else:
            print(f"[sharded] ADD from shard {shard_index}/{parts}: {shard_grouped}")
            vs.add_documents(docs)
    


def build_add_index(name: str, mode: str, input_path: str, output_path: str = None):
    if not output_path:
        output_path = f"outputs/{name}_data.json"

    load_coco(input_path=input_path, output_path=output_path)
    docs = prepare_documents(output_path, input_path)

    vs = FaissVectorStore(name)
    if mode == "build":
        vs.create_from_documents(docs)
    elif mode == "add":
        vs.add_documents(docs)
    else:
        raise ValueError("Only build or add modes exist.")

