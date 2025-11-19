from argparse import ArgumentParser
from vectorstores import FaissVectorStore
from utils import load_coco, prepare_documents
from pathlib import Path


def build_add_coco_index(name: str, mode: str, input_path: str):

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_path not found: {input_path}")
    source = input_path.name.split(".")[0]

    out_dir = Path("outputs") / "coco_captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{source}.json"

    load_coco(input_path=input_path, output_path=output_path)
    docs = prepare_documents(output_path, source)

    vs = FaissVectorStore(name)
    if mode == "build":
        added = vs.create_from_documents(docs)
        print(f"{name} vector store has been created with {added} documents.")
    elif mode == "add":
        added = vs.add_documents(docs)
        print(f"NEW {added} documents has been added to {name} vector store.")
    else:
        raise ValueError("Only build or add modes exist.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Build or add to a FAISS index from COCO data.")
    parser.add_argument("name", type=str, help="Name of the vector store.")
    parser.add_argument("mode", type=str, choices=["build", "add"], help="Mode: build a new index or add to an existing one.")
    parser.add_argument("input_path", type=str, help="Path to the input COCO data.")

    args = parser.parse_args()
    build_add_coco_index(args.name, args.mode, args.input_path)