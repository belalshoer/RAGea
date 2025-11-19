from langchain_core.documents import Document
from typing import List
import json
import tqdm
from pathlib import Path

def prepare_documents(input_path: Path, source: str) -> List[Document]:
    print(f"Preparing documents for Vector Store: ")
    docs = []

    with open(input_path, encoding = "UTF-8") as f:
        data = json.load(f)
        for img_id, captions in tqdm.tqdm(data.items()):   
            docs.append(
                Document(
                    page_content = captions["en"],
                    metadata={
                        "img_id": img_id,
                        "source": source,
                        "language": "en"
                    },
                )
            )

    return docs