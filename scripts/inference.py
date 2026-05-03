"""
inference.py
------------
Standalone inference for the fine-tuned BANKING77 intent classifier.

Required interface (as specified in the assignment):

    class IntentClassification:
        def __init__(self, model_path): ...   # loads config, tokenizer, model
        def __call__(self, message): ...      # returns predicted label string

Usage example:
    from scripts.inference import IntentClassification

    clf = IntentClassification("configs/inference.yaml")
    label = clf("I lost my credit card and need a replacement")
    print(label)   # e.g. "lost_or_stolen_card"

Or from the command line (via inference.sh):
    python scripts/inference.py --config configs/inference.yaml \
                                --message "What is my account balance?"
"""

import argparse
import os
import yaml
import torch
import warnings                                 
warnings.filterwarnings("ignore")
from unsloth import FastLanguageModel


def build_prompt(text: str) -> str:
    return f"Classify the banking intent of the following message.\nMessage: {text}\nIntent:"


class IntentClassification:
    """
    Loads a fine-tuned intent classification model and tokenizer,
    then predicts the intent label for a given input message.
    """

    def __init__(self, model_path: str):
        with open(model_path) as f:
            cfg = yaml.safe_load(f)

        checkpoint_dir = cfg["model_checkpoint"]
        label_map_path = cfg.get("label_map", os.path.join(checkpoint_dir, "label_map.yaml"))
        max_seq_length = cfg.get("max_seq_length", 128)
        load_in_4bit   = cfg.get("load_in_4bit", True)

        # Label map
        with open(label_map_path) as f:
            lm = yaml.safe_load(f)
        self.label2id = lm["label2id"]
        self.id2label = {int(k): v for k, v in lm["id2label"].items()}
        num_labels = len(self.id2label)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"[IntentClassification] Loading model from: {checkpoint_dir}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name    = checkpoint_dir,
            max_seq_length= max_seq_length,
            load_in_4bit  = load_in_4bit,
            dtype         = None,
        )

        # Classifier head
        hidden_size = self.model.config.hidden_size
        self.model.classifier = torch.nn.Linear(hidden_size, num_labels).to(self.device)
        head_path = os.path.join(checkpoint_dir, "classifier_head.pt")
        self.model.classifier.load_state_dict(
            torch.load(head_path, map_location=self.device)
        )

        FastLanguageModel.for_inference(self.model)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_seq_length = max_seq_length
        print(f"[IntentClassification] Ready on device: {self.device}")
        
    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single input message.

        Parameters
        ----------
        message : str
            Raw user message, e.g. "How do I reset my PIN?"

        Returns
        -------
        str
            Predicted intent label, e.g. "change_pin"
        """
        prompt = build_prompt(message)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**enc, output_hidden_states=True)
            hidden  = outputs.hidden_states[-1]                       # (1, T, H)
            seq_len = enc["attention_mask"].sum(dim=1) - 1            # (1,)
            pooled  = hidden[0, seq_len[0]].to(self.model.classifier.weight.dtype)
            logits  = self.model.classifier(pooled)
            pred_id = logits.argmax(dim=-1).item()

        return self.id2label[pred_id]


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run intent classification inference.")
    parser.add_argument("--config",  default="configs/inference.yaml",
                        help="Path to the inference YAML config.")
    parser.add_argument("--message", type=str, default=None,
                        help="A single message to classify. If omitted, enters interactive mode.")
    args = parser.parse_args()

    clf = IntentClassification(args.config)

    if args.message:
        label = clf(args.message)
        print(f"\nInput  : {args.message}")
        print(f"Intent : {label}")
    else:
        print("\nInteractive mode — type 'quit' to exit.\n")
        while True:
            msg = input("Message: ").strip()
            if msg.lower() in ("quit", "exit", "q"):
                break
            if msg:
                label = clf(msg)
                print(f"Intent : {label}\n")


if __name__ == "__main__":
    main()
