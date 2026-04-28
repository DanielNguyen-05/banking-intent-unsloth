import unsloth
import argparse, os, yaml, numpy as np, pandas as pd, torch, shutil
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report
from unsloth import FastLanguageModel


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_prompt(text):
    return f"Classify the banking intent of the following message.\nMessage: {text}\nIntent:"

def tokenize_fn(examples, tokenizer, label2id, max_len):
    prompts = [build_prompt(t) for t in examples["text"]]
    full_texts = [p + " " + examples["intent"][i] for i, p in enumerate(prompts)]
    enc = tokenizer(full_texts, truncation=True, max_length=max_len, padding="max_length")
    enc["labels"] = [label2id[lbl] for lbl in examples["intent"]]
    return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"eval_accuracy": accuracy_score(labels, preds)}


class ClassificationTrainer(Trainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device=None):
        batch_samples, num_items_in_batch = [], None
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        return batch_samples, num_items_in_batch

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)
        hidden  = outputs.hidden_states[-1]
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        pooled  = hidden[torch.arange(hidden.size(0)), seq_lens]
        # cast to float32 for classifier
        logits  = model.classifier(pooled.float())
        loss    = torch.nn.CrossEntropyLoss()(logits, labels)
        class Out:
            def __init__(self, loss, logits):
                self.loss = loss; self.logits = logits
        return (loss, Out(loss, logits)) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden  = outputs.hidden_states[-1]
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            pooled  = hidden[torch.arange(hidden.size(0)), seq_lens]
            logits  = model.classifier(pooled.float())
            loss    = torch.nn.CrossEntropyLoss()(logits, labels)
        return loss.detach(), logits.detach(), labels.detach()

    def create_optimizer(self):
        """Include classifier head parameters in optimizer."""
        from torch.optim import AdamW
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "classifier" not in n]
        head_params  = list(self.model.classifier.parameters())
        self.optimizer = AdamW([
            {"params": lora_params, "lr": self.args.learning_rate},
            {"params": head_params, "lr": self.args.learning_rate * 10},  # higher lr for head
        ], weight_decay=self.args.weight_decay)
        return self.optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = cfg["data"]
    mdl_cfg  = cfg["model"]
    lora_cfg = cfg["lora"]
    tr_cfg   = cfg["training"]

    with open(data_cfg["label_map"]) as f:
        lm = yaml.safe_load(f)
    label2id = lm["label2id"]
    id2label = {int(k): v for k, v in lm["id2label"].items()}
    num_labels = len(label2id)
    print(f"Num labels: {num_labels}")

    train_df = pd.read_csv(data_cfg["train_csv"])
    test_df  = pd.read_csv(data_cfg["test_csv"])
    raw_ds   = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "intent"]]),
        "test":  Dataset.from_pandas(test_df[["text",  "intent"]]),
    })

    print(f"Loading base model: {mdl_cfg['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = mdl_cfg["base_model"],
        max_seq_length= mdl_cfg["max_seq_length"],
        load_in_4bit  = mdl_cfg["load_in_4bit"],
        dtype         = None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r              = lora_cfg["r"],
        lora_alpha     = lora_cfg["lora_alpha"],
        lora_dropout   = lora_cfg["lora_dropout"],
        target_modules = lora_cfg["target_modules"],
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )

    # Add classifier head AFTER peft — keep it in float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)
    model.num_labels  = num_labels

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    max_len = mdl_cfg["max_seq_length"]
    tokenised = raw_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer, label2id, max_len),
        batched=True,
        remove_columns=["text", "intent"],
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir                  = tr_cfg["output_dir"],
        per_device_train_batch_size = tr_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size  = tr_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps = tr_cfg["gradient_accumulation_steps"],
        num_train_epochs            = tr_cfg["num_train_epochs"],
        learning_rate               = tr_cfg["learning_rate"],
        lr_scheduler_type           = tr_cfg["lr_scheduler_type"],
        warmup_steps                = 20,
        weight_decay                = tr_cfg["weight_decay"],
        optim                       = "adamw_torch",   # use standard AdamW so we can override
        fp16                        = tr_cfg["fp16"],
        bf16                        = tr_cfg["bf16"],
        logging_steps               = tr_cfg["logging_steps"],
        eval_strategy               = tr_cfg["eval_strategy"],
        save_strategy               = tr_cfg["save_strategy"],
        load_best_model_at_end      = tr_cfg["load_best_model_at_end"],
        metric_for_best_model       = tr_cfg["metric_for_best_model"],
        seed                        = tr_cfg["seed"],
        report_to                   = "none",
    )

    trainer = ClassificationTrainer(
        model         = model,
        args          = training_args,
        train_dataset = tokenised["train"],
        eval_dataset  = tokenised["test"],
        data_collator = collator,
        compute_metrics = compute_metrics,
    )

    print("Starting training ...")
    trainer.train()

    print("\nEvaluating on test set ...")
    pred_output = trainer.predict(tokenised["test"])
    preds  = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids
    acc    = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=[id2label[i] for i in range(num_labels)]))

    best_dir = os.path.join(tr_cfg["output_dir"], "best_model")
    os.makedirs(best_dir, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    torch.save(model.classifier.state_dict(), os.path.join(best_dir, "classifier_head.pt"))
    shutil.copy(data_cfg["label_map"], os.path.join(best_dir, "label_map.yaml"))
    print(f"\nModel saved to: {best_dir}")


if __name__ == "__main__":
    main()