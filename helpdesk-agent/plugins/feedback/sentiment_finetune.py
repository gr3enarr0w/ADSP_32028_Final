"""LoRA and full fine-tune helpers for M2 sentiment (ANTSE-326)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from plugins.feedback.pipeline import score_to_intensity
from plugins.feedback.sentiment_agreement import (
    AGREEMENT_THRESHOLD,
    compute_agreement_rate,
    csat_to_training_label,
    spearman_csat_sentiment,
)

log = logging.getLogger(__name__)

BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FINETUNED_DIR = REPO_ROOT / "models" / "sentiment_finetuned"
MODEL_RUNS_DIR = REPO_ROOT / "models" / "sentiment_runs"

LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["query", "value"]


def stratified_train_val_test_split(
    ticket_keys: list[str],
    categories: list[str],
    *,
    train_size: float = 0.8,
    random_state: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train, val, test) index lists using a stratified 80/10/10 split.

    The remaining 20% after the train split is divided equally between val and
    test, giving 10% each.  Stratification is by category at both split points.
    """
    indices = np.arange(len(ticket_keys))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1.0 - train_size,
        stratify=categories,
        random_state=random_state,
    )
    temp_categories = [categories[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # half of the 20% temp → 10% val, 10% test
        stratify=temp_categories,
        random_state=random_state,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def evaluate_split(
    csat_scores: list[int],
    sentiment_scores: list[float],
    intensities: list[str],
) -> dict:
    """Binary agreement + Spearman on a held-out split."""
    agreement = compute_agreement_rate(csat_scores, intensities)
    rho, pval = spearman_csat_sentiment(csat_scores, sentiment_scores)
    return {
        "agreement": agreement,
        "spearman_rho": rho,
        "spearman_p": pval,
    }


def intensities_from_neg_scores(
    neg_scores: list[float],
    high_threshold: float = 0.6,
) -> list[str]:
    return [score_to_intensity(float(s), high_threshold) for s in neg_scores]


def _training_args(output_dir: Path, epochs: int = 3):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        use_cpu=not _has_accelerator(),
    )


def _has_accelerator() -> bool:
    try:
        import torch

        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        return False


def _pipeline_device() -> int:
    """Return a transformers pipeline device index.

    Keep inference CPU-safe on Apple Silicon where pipeline CUDA indices
    are not valid for MPS-only hosts.
    """
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def train_sentiment_classifier(
    texts: list[str],
    labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    *,
    output_dir: Path,
    use_lora: bool = True,
    epochs: int = 3,
):
    """Train cardiffnlp with optional LoRA; saves model to output_dir."""
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        DataCollatorWithPadding,
    )

    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        log.info("LoRA adapters attached (r=%s, alpha=%s)", LORA_R, LORA_ALPHA)
    else:
        log.info("Full fine-tune — all parameters trainable")

    def _encode(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )
        enc["labels"] = [label2id[l] for l in batch["label"]]
        return enc

    train_ds = Dataset.from_dict({"text": texts, "label": labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    train_ds = train_ds.map(_encode, batched=True, remove_columns=["text", "label"])
    val_ds = val_ds.map(_encode, batched=True, remove_columns=["text", "label"])

    args = _training_args(output_dir / "checkpoints", epochs=epochs)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    if use_lora:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    log.info("Saved fine-tuned model to %s", output_dir)


def predict_neg_scores(texts: list[str], model_dir: Path) -> list[float]:
    """Return NEGATIVE-class probability per text from a saved model directory."""
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")

    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "sentiment-analysis",
        model=str(model_dir),
        tokenizer=str(model_dir),
        top_k=None,
        device=_pipeline_device(),
    )
    scores: list[float] = []
    for text in texts:
        truncated = (text or "").strip()[:1500]
        if not truncated:
            scores.append(0.0)
            continue
        raw = pipe([truncated])[0]
        label_scores = {item["label"].upper(): item["score"] for item in raw}
        neg = label_scores.get("NEGATIVE", label_scores.get("LABEL_0", 0.0))
        scores.append(float(neg))
    return scores


def run_finetune_strategy(
    records: list[dict],
    *,
    use_lora: bool,
    high_threshold: float = 0.6,
    random_state: int = 42,
) -> dict:
    """80/10/10 train, evaluate binary agreement on test split."""
    keys = [r["ticket_key"] for r in records]
    categories = [r["category"] for r in records]
    texts = [r["text"] for r in records]
    csat_scores = [int(r["csat_score"]) for r in records]

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        keys, categories, random_state=random_state
    )

    train_texts = [texts[i] for i in train_idx]
    train_labels = [csat_to_training_label(csat_scores[i]) for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [csat_to_training_label(csat_scores[i]) for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    test_csat = [csat_scores[i] for i in test_idx]

    tag = "lora" if use_lora else "full"
    out_dir = MODEL_RUNS_DIR / tag
    train_sentiment_classifier(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        output_dir=out_dir,
        use_lora=use_lora,
    )

    neg_scores = predict_neg_scores(test_texts, out_dir)
    intensities = intensities_from_neg_scores(neg_scores, high_threshold)
    metrics = evaluate_split(test_csat, neg_scores, intensities)
    metrics["strategy"] = tag
    metrics["deployed"] = metrics["agreement"] >= AGREEMENT_THRESHOLD
    if metrics["deployed"]:
        # Promote winning checkpoint to canonical path
        import shutil

        if FINETUNED_DIR.exists():
            shutil.rmtree(FINETUNED_DIR, ignore_errors=True)
        FINETUNED_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out_dir, FINETUNED_DIR)
        log.info("Deployed model to %s", FINETUNED_DIR)
    return metrics
