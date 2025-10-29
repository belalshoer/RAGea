from pipelines import build_add_index
from argparse import ArgumentParser
import sys

def main():
    parser = ArgumentParser(description="Build or add to a FAISS index from COCO data.")
    parser.add_argument("name", type=str, help="Name of the vector store.")
    parser.add_argument("mode", type=str, choices=["build", "add"], help="Mode: build a new index or add to an existing one.")
    parser.add_argument("input_path", type=str, help="Path to the input COCO data.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the processed data (optional).")

    args = parser.parse_args()
    build_add_index(args.name, args.mode, args.input_path, args.output_path)


if __name__ == "__main__":
    main()