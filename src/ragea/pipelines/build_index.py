from vectorstores import FaissVectorStore
from utils import load_coco, prepare_documents
from pathlib import Path


def build_add_index(name: str, mode: str, input_path: str, output_path: str = None):
    if not output_path:
        output_path = Path("src") / "ragea" / "outputs" / f"{name}_data.json"
    else:
        output_path = Path(output_path)

    load_coco(input_path=input_path, output_path=output_path)
    source = input_path.split("/")[-1]
    docs = prepare_documents(output_path, source)

    vs = FaissVectorStore(name)
    if mode == "build":
        vs.create_from_documents(docs)
    elif mode == "add":
        vs.add_documents(docs)
    else:
        raise ValueError("Only build or add modes exist.")

