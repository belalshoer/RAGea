from langchain_core.documents import Document
import json

def prepare_documents(input_path, source):
    docs = []

    with open(input_path, encoding = "UTF-8") as f:
        data = json.load(f)

        for img_id, captions in data.items():   
            for lang, caption in captions.items():
                docs.append(
                    Document(
                        page_content = caption,
                        metadata={
                            "img_id": img_id,
                            "source": source.split("/")[-1],
                            "language": lang
                        },
                    )
                )

    return docs