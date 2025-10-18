import argparse
from typing import Any, Dict, List, Optional
from datasets import load_dataset
import pandas as pd

from ragea.generation.pangea import caption_images, caption_image
from ragea.pipelines.metrics import evaluate_captions  # uses BLEU@4, ROUGE-L, ChrF++, BERTScore

LANG_SPLITS = [
    "ar","bn","cs","da","de","el","en","es","fa","fi","fil","fr","hi","hr","hu",
    "id","it","he","ja","ko","mi","nl","no","pl","pt","quz","ro","ru","sv","sw",
    "te","th","tr","uk","vi","zh"
]

LANG_NAME = {
    "ar":"Arabic","bn":"Bengali","cs":"Czech","da":"Danish","de":"German","el":"Greek",
    "en":"English","es":"Spanish","fa":"Persian","fi":"Finnish","fil":"Filipino",
    "fr":"French","hi":"Hindi","hr":"Croatian","hu":"Hungarian","id":"Indonesian",
    "it":"Italian","he":"Hebrew","ja":"Japanese","ko":"Korean","mi":"Māori","nl":"Dutch",
    "no":"Norwegian","pl":"Polish","pt":"Portuguese","quz":"Quechua","ro":"Romanian",
    "ru":"Russian","sv":"Swedish","sw":"Swahili","te":"Telugu","th":"Thai","tr":"Turkish",
    "uk":"Ukrainian","vi":"Vietnamese","zh":"Chinese"
}

def caption_backend_pangea(images: List[Any], user_prompt: str) -> List[str]:
    """Use ragea.generation.pangea; prefer true batched path if available, fallback to sequential on runtime errors."""
    
    try:
        return caption_images(image_paths=images, batch_size=4, user_prompt=user_prompt)
    except Exception:
        # Fallback: sequential (keep try/except here for robustness at runtime; imports at top are clean)
        return [caption_image(image_path=img, user_prompt=user_prompt) for img in images]

def caption_backend_pipeline(images: List[Any], user_prompt: str) -> List[str]:
    """MOCK backend (intentionally left unimplemented)."""
    raise NotImplementedError("pipeline backend is a mock. Use --backend pangea for now.")

def eval_split(
    lang: str,
    base_prompt: str,
    backend: str,
    no_lang_hint: bool,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    ds = load_dataset("neulab/PangeaBench-xm100", split=lang)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    images = [ex["image"] for ex in ds]       # PIL.Image.Image
    image_ids = [ex["image_id"] for ex in ds]
    refs = [ex["caption"] for ex in ds]

    # Per-split prompt; by default, ask for the split's language.
    user_prompt = base_prompt.strip()
    if not no_lang_hint:
        lang_name = LANG_NAME.get(lang, lang)
        user_prompt = f"{user_prompt}\nWrite the caption in {lang_name}."

    if backend == "pangea":
        preds = caption_backend_pangea(images, user_prompt)
    elif backend == "pipeline":
        preds = caption_backend_pipeline(images, user_prompt)  # raises NotImplementedError
    else:
        raise ValueError(f"Unknown backend: {backend}")

    metrics = evaluate_captions(refs, preds, lang=lang)
    return {
        "lang": lang,
        "n": len(ds),
        "metrics": metrics,
        "image_ids": image_ids,
        "references": refs,
        "predictions": preds,
        "prompt_used": user_prompt,
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate Pangea on XM-100 (consistent multilingual metrics).")
    ap.add_argument("--backend", choices=["pangea", "pipeline"], default="pangea",
                    help="Use pangea functions or (mock) HF pipeline.")
    ap.add_argument("--prompt", default="Caption the image in one concise sentence.",
                    help="Base user prompt for captioning.")
    ap.add_argument("--langs", nargs="+", default=["all"],
                    help='Language splits to evaluate (e.g., "en fr de") or "all".')
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Optional limit per language (<=100).")
    ap.add_argument("--no_lang_hint", action="store_true",
                    help="If set, do not append 'Write the caption in <Language>.' to the prompt.")
    ap.add_argument("--save_csv", type=str, default=None,
                    help="Optional path to write CSV of results.")

    args = ap.parse_args()
    langs = LANG_SPLITS if (len(args.langs) == 1 and args.langs[0].lower() == "all") else args.langs

    all_rows = []
    for lang in langs:
        print(f"\n=== Evaluating split: {lang} ===", flush=True)
        res = eval_split(
            lang=lang,
            base_prompt=args.prompt,
            backend=args.backend,
            no_lang_hint=args.no_lang_hint,
            max_samples=args.max_samples,
        )
        m = res["metrics"]
        lang_full = LANG_NAME.get(lang, lang)

        line = (
        f"[metrics] {lang} ({lang_full}) backend:{args.backend}: "
        f"BLEU@4={m.get('BLEU@4', float('nan')):.4f}  "
        f"ROUGE-L={m.get('ROUGE-L', float('nan')):.4f}  "
        f"ChrF++={m.get('ChrF++', float('nan')):.4f}  "
        f"BERT-F1={m.get('bert_F1', float('nan')):.4f}"
)

        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        for iid, ref, pred in zip(res["image_ids"], res["references"], res["predictions"]):
            all_rows.append({"lang": lang, "image_id": iid, "reference": ref, "prediction": pred})
        if args.save_csv:
            csv= lang+args.save_csv
            pd.DataFrame(all_rows).to_csv(csv, index=False)
            print(f"[info] wrote CSV to {args.save_csv}")

    if args.save_csv:
        pd.DataFrame(all_rows).to_csv(args.save_csv, index=False)
        print(f"[info] wrote CSV to {args.save_csv}")

if __name__ == "__main__":
    main()
