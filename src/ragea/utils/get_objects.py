from typing import Sequence
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo11s.pt")

def _extract_objects_from_result(image: Image.Image, r) -> list[dict[str, object]]:
    """
    Helper to turn a single Ultralytics result into your desired list of dicts.
    """
    W, H = image.size
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy  = boxes.xyxy.cpu().numpy()                  # [N,4]
    clses = boxes.cls.cpu().numpy().astype(int)       # [N]
    name_map = r.names                                 # {id: name}

    objects = []
    for (x1, y1, x2, y2), c in zip(xyxy, clses):
        # clamp to image bounds and cast to int
        x1 = int(max(0, min(W, x1)))
        y1 = int(max(0, min(H, y1)))
        x2 = int(max(0, min(W, x2)))
        y2 = int(max(0, min(H, y2)))
        if x2 <= x1 or y2 <= y1:
            continue  # skip degenerate boxes just in case

        crop = image.crop((x1, y1, x2, y2))
        objects.append({
            "name": name_map.get(c, str(c)),
            "crop": crop
        })
    return objects


def get_objects(image: any, min_confidence: float, model=model) -> list[dict[str, object]]:
    """
    Single-image version (unchanged behavior).
    Accepts a PIL image or a file path. Returns a flat list of detections.
    """
    if   isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise TypeError("the function expects an image path or a PIL image")

    results = model.predict(pil_img, conf=min_confidence, verbose=False)
    if not results:
        return []
    return _extract_objects_from_result(pil_img, results[0])


def get_objects_batch(
    images: Sequence[Image.Image],
    min_confidence: float,
    model=model,
    batch_size: int | None = None,
) -> list[list[dict[str, object]]]:
    """
    Batch version.
    Expects a sequence of PIL images and returns a list of per-image detection lists.

    Args:
        images: Sequence of PIL.Image.Image
        min_confidence: confidence threshold for YOLO
        model: loaded Ultralytics YOLO model
        batch_size: optional internal batching hint for Ultralytics (e.g., 8 or 16)

    Returns:
        detections_per_image: list where each element is the same structure as `get_objects`,
        i.e., List[Dict[str, object]] with keys {"name", "crop"} for the corresponding input image.
    """
    if not isinstance(images, (list, tuple)):
        raise TypeError("images must be a list or tuple of PIL.Image.Image")

    if not all(isinstance(im, Image.Image) for im in images):
        raise TypeError("each item in images must be a PIL.Image.Image")

    if len(images) == 0:
        return []

    # Run a single batched prediction call; Ultralytics keeps order aligned with inputs.
    predict_kwargs = dict(conf=min_confidence, verbose=False)
    if batch_size is not None:
        predict_kwargs["batch"] = batch_size

    results = model.predict(images, **predict_kwargs)

    # Safety: results length should match input length
    if not results or len(results) != len(images):
        # Fallback: produce empty lists for all images if something unexpected happened
        return [[] for _ in images]

    detections_per_image = []
    for pil_img, r in zip(images, results):
        detections_per_image.append(_extract_objects_from_result(pil_img, r))

    return detections_per_image


if __name__ == "__main__":
    # Example usage
    image_path = "example.png"

    # Single image (path)
    objs = get_objects(image_path, 0.5, model=model)
    print("path_trial:")
    for obj in objs:
        print(obj["name"])

    # Single image (PIL)
    img = Image.open(image_path)
    objs = get_objects(img, 0.5, model=model)
    print("PIL_trial:")
    for obj in objs:
        print(obj["name"])

    # Batch (PIL list)
    imgs = [Image.open(image_path), Image.open(image_path)]
    batch_objs = get_objects_batch(imgs, 0.5, model=model, batch_size=8)
    print("BATCH_trial:")
    for i, dets in enumerate(batch_objs):
        print(f"Image {i}: {[d['name'] for d in dets]}")
