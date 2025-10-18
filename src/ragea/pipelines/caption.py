from argparse import ArgumentParser
import os
import sys
import pandas as pd


from ragea.generation.pangea import caption_image, caption_images
   

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTS


def list_images_in_dir(directory: str):
    # Non-recursive; change to os.walk(...) if you want recursion
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if is_image_file(os.path.join(directory, f))
    ]
    files.sort()
    return files


def main():
    parser = ArgumentParser(description="Caption an image or all images in a directory.")
    parser.add_argument("--path", required=True, help="Path to an image file or a directory of images.")
    parser.add_argument("-p", "--prompt", help="Optional user prompt to guide captions.")
    parser.add_argument("-c", "--count", type=int, help="Only process the first N images in a directory.")
    args = parser.parse_args()

    target = args.path
    if not os.path.exists(target):
        print(f"[error] Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    # File case
    if os.path.isfile(target):
        if not is_image_file(target):
            print(f"[error] Not a supported image file: {target}", file=sys.stderr)
            sys.exit(1)
        if args.prompt:
            caption = caption_image(image_path=target, user_prompt=args.prompt)
        else:
            caption = caption_image(image_path=target)
        print(f"{target}\t{caption}")
        return

    # Directory case
    if os.path.isdir(target):
        images = list_images_in_dir(target)
        if not images:
            print("[warn] No images found in directory.", file=sys.stderr)
            return
        if args.count is not None and args.count > 0:
            images = images[: args.count]

        batch_size = min(32, len(images))

    kwargs = {"image_paths": images, "batch_size": batch_size}
    if args.prompt:
        kwargs["user_prompt"] = args.prompt

    captions = caption_images(**kwargs)

    # Sanity check (optional)
    assert len(captions) == len(images), "Mismatch between images and captions!"

    df = pd.DataFrame({
        "filename": images,
        "captions": captions,
    })

    df.to_csv("captions.csv", index=False)
    return

    print(f"[error] Unsupported path type: {target}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
