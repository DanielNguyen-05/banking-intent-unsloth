"""
preprocess_data.py
------------------
Downloads full BANKING77, preprocesses it, and saves train/test CSVs.

Usage:
    python scripts/preprocess_data.py --config configs/train.yaml
"""

import argparse
import os
import re
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s,.?!'-]", "", text)
    return text.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg      = cfg["data"]
    test_size     = data_cfg.get("test_size", 0.2)
    random_seed   = data_cfg.get("random_seed", 42)
    train_out     = data_cfg.get("train_csv",  "sample_data/train.csv")
    test_out      = data_cfg.get("test_csv",   "sample_data/test.csv")
    label_map_out = data_cfg.get("label_map",  "sample_data/label_map.yaml")

    os.makedirs("sample_data", exist_ok=True)

    print("[1/4] Loading full BANKING77 from GitHub CSV ...")
    train_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    test_url  = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
    df_train_full = pd.read_csv(train_url)
    df_test_full  = pd.read_csv(test_url)
    df_all = pd.concat([df_train_full, df_test_full], ignore_index=True)
    df_all = df_all.rename(columns={"category": "intent"})
    print(f"  Total samples : {len(df_all)}")
    print(f"  Total intents : {df_all['intent'].nunique()}")

    print("[2/4] Preprocessing ...")
    unique_intents = sorted(df_all["intent"].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(unique_intents)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    df_all["text"]     = df_all["text"].apply(normalize_text)
    df_all["label_id"] = df_all["intent"].map(label2id)

    print(f"[3/4] Splitting train/test (test_size={test_size}) ...")
    train_df, test_df = train_test_split(
        df_all[["text", "intent", "label_id"]],
        test_size=test_size,
        random_state=random_seed,
        stratify=df_all["label_id"],
    )

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out,   index=False)
    with open(label_map_out, "w") as f:
        yaml.dump({"label2id": label2id, "id2label": id2label}, f)

    print(f"[4/4] Done.")
    print(f"  Train samples : {len(train_df)}")
    print(f"  Test  samples : {len(test_df)}")
    print(f"  Num intents   : {len(unique_intents)}")
    print(f"  Saved -> {train_out}, {test_out}, {label_map_out}")


if __name__ == "__main__":
    main()