from vectorstores import FaissVectorStore
from utils import load_coco, prepare_documents
from pathlib import Path


def build_add_index(name: str, mode: str, input_path: str, output_path: str = None):

    source = input_path.split("/")[-1]
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input_path not found: {input_path}")


    if not output_path:
        output_path = Path("outputs") / f"{name}_vs_{source}_data.json"
    else:
        output_path = Path(output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"output_path not found: {output_path}")

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
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the processed data (optional).")

    args = parser.parse_args()
    build_add_index(args.name, args.mode, args.input_path, args.output_path)