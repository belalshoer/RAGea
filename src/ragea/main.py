from pipelines import build_add_index
from vectorstores import FaissVectorStore
from PIL import Image


# build_add_index_sharded(
#     name="english_coco_index",
#     input_path="/home/belal.shoer/Desktop/RAGea/COCO-35L/english_only.jsonl",
#     parts=64,
#     overwrite=True,   
# )
# build_add_index("english_coco_index3", "build", "/home/belal.shoer/Desktop/RAGea/COCO-35L/english_only.jsonl")

vs = FaissVectorStore("english_coco_index3")
img = Image.open("/home/belal.shoer/Desktop/VtoL/nanoVLM/chihuahua-puppy-grass-squeaky-bone-toy-58273.webp")
print(vs.retrieve(img))
