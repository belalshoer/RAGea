from vectorstores import FaissVectorStore
from utils import load_coco, prepare_documents

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
        

    raise ValueError("Only build or add modes exist.")

