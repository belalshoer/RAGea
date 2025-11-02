from argparse import ArgumentParser
import os
import sys
import pandas as pd
from generation import Captioner
   
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_images_in_dir(directory: str):
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if is_image_file(os.path.join(directory, f))
    ]
    files.sort()
    return files


def caption(path, prompt, count):

    captioner = Captioner()

    # File case
    if os.path.isfile(target):
        if not is_image_file(target):
            print(f"[error] Not a supported image file: {target}", file=sys.stderr)
            sys.exit(1)
        if prompt:
            caption = captioner.caption_image(image=target, user_prompt=prompt)
        else:
            caption = captioner.caption_image(image=target)
        print(f"{target}\t{caption}")
        return

    # Directory case
    if os.path.isdir(target):
        images = list_images_in_dir(target)
        if not images:
            print("[warn] No images found in directory.", file=sys.stderr)
            return
        if count is not None and count > 0:
            images = images[: count]

    batch_size = min(4, len(images))

    kwargs = {"images": images, "batch_size": batch_size}
    if prompt:
        kwargs["user_prompt"] = prompt

    captions = captioner.caption_images(**kwargs)

    assert len(captions) == len(images), "Mismatch between images and captions!"

    df = pd.DataFrame({
        "filename": [os.path.basename(image) for image in images],
        "captions": captions,
    })

    os.makedirs("outputs", exist_ok=True)

    df.to_csv("outputs/captions.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Caption an image or all images in a directory.")
    parser.add_argument("--path", required=True, help="Path to an image file or a directory of images.")
    parser.add_argument("-p", "--prompt", help="Optional user prompt to guide captions.")
    parser.add_argument("-c", "--count", type=int, help="Only process the first N images in a directory.")
    args = parser.parse_args()

    target = args.path
    if not os.path.exists(target):
        print(f"[error] Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    caption(args.path, args.prompt, args.count)