import argparse
from typing import Any, Dict, List, Optional, Sequence
from datasets import load_dataset
from datetime import datetime
import pandas as pd
from generation import Captioner
from pathlib import Path
from utils import preprocess_similar_captions_xm100, get_similar_captions_xm100, evaluate_captions
from vectorstores import FaissVectorStore
import os
import tqdm

LANG_SPLITS = [
    "ar","bn","cs","da","de","el","en","es","fa","fi","fil","fr","hi","hr","hu",
    "id","it","he","ja","ko","mi","nl","no","pl","pt","ro","ru","sv","sw",
    "te","th","tr","uk","vi","zh"
]

LANG_NAME = {
    "ar":"Arabic","bn":"Bengali","cs":"Czech","da":"Danish","de":"German","el":"Greek",
    "en":"English","es":"Spanish","fa":"Persian","fi":"Finnish","fil":"Filipino",
    "fr":"French","hi":"Hindi","hr":"Croatian","hu":"Hungarian","id":"Indonesian",
    "it":"Italian","he":"Hebrew","ja":"Japanese","ko":"Korean","mi":"Māori","nl":"Dutch",
    "no":"Norwegian","pl":"Polish","pt":"Portuguese","ro":"Romanian",
    "ru":"Russian","sv":"Swedish","sw":"Swahili","te":"Telugu","th":"Thai","tr":"Turkish",
    "uk":"Ukrainian","vi":"Vietnamese","zh":"Chinese"
}

print("Loading Pangea...")
captioner = Captioner()

def caption_backend_pangea(images: List[Any], lang: str, similar_captions: Sequence[List[str]] = None) -> List[str]:
    try:
        return captioner.caption_images(images=images, lang=lang, similar_captions=similar_captions)
    except Exception:
        if similar_captions:
            return [captioner.caption_image(image=img, lang=lang, similar_captions=captions) 
                    for img, captions in tqdm.tqdm(zip(images, similar_captions))]
        else:
            return [captioner.caption_image(image=img, lang=lang) 
                    for img in tqdm.tqdm(images)]


def eval_xm100_split(
    lang: str,
    vector_store_name: str = None,
    k: int = None,
    search_type: str = None
) -> Dict[str, Any]:
    print(f"Loading Dataset for {LANG_NAME.get(lang, lang)}...")
    ds = load_dataset("neulab/PangeaBench-xm100", split=lang)

    images = [ex["image"] for ex in ds]     
    image_ids = [ex["image_id"] for ex in ds]
    refs = [ex["caption"] for ex in ds]

    lang_name = LANG_NAME.get(lang, lang)

    similar_captions = None
    if vector_store_name:   
        similar_captions = get_similar_captions_xm100(vector_store_name, lang, k, search_type=search_type)
    preds = caption_backend_pangea(images=images, similar_captions=similar_captions, lang=lang_name)

    metrics = evaluate_captions(refs, preds, lang=lang)
    return {
        "lang": lang,
        "n": len(ds),
        "metrics": metrics,
        "image_ids": image_ids,
        "references": refs,
        "predictions": preds,
    }

def evaluate_xm100(
    vector_store_name: str = None,
    k: int = None,
    langs: List[str] = "all",
    search_type: str = None,
    run_name: str = None
):

    langs = LANG_SPLITS if (len(langs) == 1 and langs[0].lower() == "all") else langs

    out_dir = Path("outputs") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / (run_name or f"run-{stamp}")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except Exception as e:
        raise("You already have an experiment with this name!")


    if vector_store_name:
        vs = FaissVectorStore(vector_store_name)
        similar_captions_path = Path("outputs") / "similar_captions" / f"{vector_store_name}" / f"top_{k} captoins using {search_type}.json"
        if not similar_captions_path.exists():
            preprocess_similar_captions_xm100(vs, k, search_type)
        
    all_rows = []
    for lang in langs:
        print(f"\n=== Evaluating split: {LANG_NAME.get(lang, lang)} ===", flush=True)

        res = eval_xm100_split(
            lang=lang,
            vector_store_name=vector_store_name,
            k=k,
            search_type=search_type
        )

        m = res["metrics"]
        lang_full = LANG_NAME.get(lang, lang)

        line = (
            f"[metrics] {lang} ({lang_full}) : "
            f"BLEU@4={m.get('BLEU@4', float('nan')):.4f}  "
            f"ROUGE-L={m.get('ROUGE-L', float('nan')):.4f}  "
            f"ChrF++={m.get('ChrF++', float('nan')):.4f}  "
            f"BERT-F1={m.get('bert_F1', float('nan')):.4f}"
        )

        with open(run_dir / f"metrics.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        for iid, ref, pred in zip(res["image_ids"], res["references"], res["predictions"]):
            all_rows.append({"lang": lang, "image_id": iid, "reference": ref, "prediction": pred})


    full_path = run_dir / f"results.csv"
    pd.DataFrame(all_rows).to_csv(full_path, index=False)
    print(f"[info] wrote CSV to {full_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate Pangea on XM-100 (consistent multilingual metrics).")
    ap.add_argument("--vector_store_name", default = None,
                    help="The name of the vector store to retrieve from. If None, no retrieval will be done.")
    ap.add_argument("--k", type = int, default = 4,
                    help="Number of captions to retrieve.")
    ap.add_argument("--search_type", type=str, default=None, choices=["similarity", "mmr"],
                    help="Method of search. similarity or mmr")
    ap.add_argument("--langs", nargs="+", default=["all"],
                    help='Language splits to evaluate (e.g., "en fr de") or "all".')
    ap.add_argument("--run_name", type=str, default=None,
                    help="Name of the experiment.")

    args = ap.parse_args()

    payload = {k: v for k, v in vars(args).items() if v is not None}
    evaluate_xm100(**payload)