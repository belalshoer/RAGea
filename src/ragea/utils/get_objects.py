from ultralytics import YOLO
from PIL import Image
model= YOLO("yolo11s.pt")

def get_objects(image: any,min_confidence:float,model=model):
    """
    Returns a list of detections. Each item has:
      - 'name': class name (str)
      - 'crop': PIL.Image.Image cropped to the bbox
    """
    if   isinstance(image,Image.Image):
        pass
    elif isinstance(image,str):
        image=Image.open(image)
    else:
        raise TypeError("the function expects an image path or a PIL image")
    W, H = image.size
    results = model.predict(image, conf=min_confidence, verbose=False)
    if not results:
        return []

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return []

    # Fetch tensors and map to CPU/NumPy for easy iteration
    xyxy = boxes.xyxy.cpu().numpy()       # shape: [N,4]
    clses = boxes.cls.cpu().numpy().astype(int)  # shape: [N]
    name_map = r.names  # {class_id: class_name}

    objects=[]
    for (x1, y1, x2, y2), c in zip(xyxy, clses):
        # clamp & cast to int pixel coords
        x1, y1, x2, y2 = map(int, (max(0, min(W, x1)),
                                   max(0, min(H, y1)),
                                   max(0, min(W, x2)),
                                   max(0, min(H, y2))))
        crop = image.crop((x1, y1, x2, y2))
        objects.append({
            "name": name_map.get(c, str(c)),
            "crop": crop
        })
    return objects


if __name__=="__main__":

    
    #try function
    image_path="example.png"
    #try path
    objects=get_objects(image_path,0.5,model=model)
    print("path_trial:")
    for obj in objects:
        print(obj["name"])
            

    image=Image.open(image_path)
    objects=get_objects(image,0.5,model=model)
    print("PIL_trial:")
    for obj in objects:
        print(obj["name"])
       

