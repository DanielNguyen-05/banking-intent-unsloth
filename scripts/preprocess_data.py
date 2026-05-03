import argparse
import os
import re
import yaml
import numpy as np
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

    data_cfg           = cfg["data"]
    test_size          = data_cfg.get("test_size", 0.2)
    random_seed        = data_cfg.get("random_seed", 42)
    num_intents        = data_cfg.get("num_intents", 40)
    samples_per_intent = data_cfg.get("samples_per_intent", 100)
    train_out          = data_cfg.get("train_csv",  "sample_data/train.csv")
    test_out           = data_cfg.get("test_csv",   "sample_data/test.csv")
    label_map_out      = data_cfg.get("label_map",  "sample_data/label_map.yaml")

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

    print(f"\nSampling dataset (num_intents={num_intents}, max_samples/intent={samples_per_intent}) ...")
    # Đặt seed để đảm bảo tính tái lập (reproducibility)
    np.random.seed(random_seed)
    
    # 1. Lấy ngẫu nhiên 'num_intents' từ danh sách các intents
    unique_intents = df_all['intent'].unique()
    sampled_intents = np.random.choice(unique_intents, min(num_intents, len(unique_intents)), replace=False)
    df_subset = df_all[df_all['intent'].isin(sampled_intents)]
    
    # 2. Lấy tối đa 'samples_per_intent' mẫu cho mỗi intent đã chọn
    sampled_parts = []

    for intent, group in df_subset.groupby("intent"):
        sampled_group = group.sample(
            n=min(len(group), samples_per_intent),
            random_state=random_seed
        )
        sampled_parts.append(sampled_group)

    df_sampled = pd.concat(sampled_parts, ignore_index=True)
    
    print(f"  Sampled samples : {len(df_sampled)}")
    print(f"  Sampled intents : {df_sampled['intent'].nunique()}\n")

    print("[2/4] Preprocessing ...")
    unique_intents_sampled = sorted(df_sampled["intent"].unique())
    label2id = {lbl: idx for idx, lbl in enumerate(unique_intents_sampled)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    df_sampled["text"]     = df_sampled["text"].apply(normalize_text)
    df_sampled["label_id"] = df_sampled["intent"].map(label2id)

    print(f"[3/4] Splitting train/test (test_size={test_size}) ...")
    train_df, test_df = train_test_split(
        df_sampled[["text", "intent", "label_id"]],
        test_size=test_size,
        random_state=random_seed,
        stratify=df_sampled["label_id"],
    )

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out,   index=False)
    with open(label_map_out, "w") as f:
        yaml.dump({"label2id": label2id, "id2label": id2label}, f)

    print(f"[4/4] Done.")
    print(f"  Train samples : {len(train_df)}")
    print(f"  Test  samples : {len(test_df)}")
    print(f"  Num intents   : {len(unique_intents_sampled)}")
    print(f"  Saved -> {train_out}, {test_out}, {label_map_out}")


if __name__ == "__main__":
    main()