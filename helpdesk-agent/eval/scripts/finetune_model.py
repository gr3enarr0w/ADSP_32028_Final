"""Two-stage fine-tuning for ticket→article retrieval.

Stage 1 — TSDAE (unsupervised):
    DenoisingAutoEncoderLoss on all ticket + article texts as unlabeled corpus.
    Teaches the model domain-specific language without labels.
    1 epoch, batch_size=16.

Stage 2 — MNRL (supervised):
    MultipleNegativesRankingLoss on (anchor, positive) training pairs
    from build_training_data.py output.
    3 epochs, batch_size=32, warmup_ratio=0.1, lr=2e-5, eval every 100 steps.

Supports MPS (Apple Silicon) acceleration when available.

Usage:
    python -m eval.scripts.finetune_model --model minilm
    python -m eval.scripts.finetune_model --model mpnet
    python -m eval.scripts.finetune_model --model minilm --skip-tsdae
    python -m eval.scripts.finetune_model --model minilm --tsdae-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.compare_strategies import MODELS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKETS_FILE = DATA_DIR / "jiraconfsd_all.json"
ARTICLES_FILE = DATA_DIR / "confluence_articles.json"
MODELS_DIR = DATA_DIR / "models"


def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_corpus() -> list[str]:
    """Load all ticket and article texts as an unlabeled corpus for TSDAE."""
    from eval.loaders.jiraconfsd import _ticket_text

    tickets = json.loads(TICKETS_FILE.read_text())
    articles = json.loads(ARTICLES_FILE.read_text())

    texts = []
    for t in tickets:
        text = _ticket_text(t)
        if len(text) > 20:
            texts.append(text)

    for a in articles:
        title = a.get("title", "")
        body = a.get("body", "")
        text = title
        if body and len(body) > 30:
            text += ". " + body[:500 - len(title) - 2]
        if len(text) > 20:
            texts.append(text[:500])

    return texts


def _load_training_pairs(split: str) -> tuple[list[str], list[str]]:
    path = DATA_DIR / f"training_pairs_{split}.json"
    if not path.exists():
        print(f"Missing {path}. Run: python -m eval.scripts.build_training_data",
              file=sys.stderr)
        sys.exit(1)

    pairs = json.loads(path.read_text())
    anchors = [p["anchor"] for p in pairs]
    positives = [p["positive"] for p in pairs]
    return anchors, positives


def run_tsdae(model_name: str, output_dir: Path, batch_size: int = 16,
              epochs: int = 1):
    """Stage 1: TSDAE unsupervised pre-training."""
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset

    model_id = MODELS[model_name]["id"]
    device = _detect_device()
    print(f"\n{'='*60}")
    print(f"  STAGE 1: TSDAE — {model_id}")
    print(f"  Device: {device} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"{'='*60}\n")

    model = SentenceTransformer(model_id, device=device)

    corpus = _load_corpus()
    print(f"Corpus size: {len(corpus)} texts")

    train_dataset = DenoisingAutoEncoderDataset(corpus)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True)

    # MPNet doesn't have a CausalLM head, so it can't serve as its own
    # TSDAE decoder.  Use a separate bert-base-uncased decoder instead.
    from transformers import AutoModelForCausalLM
    try:
        AutoModelForCausalLM.from_pretrained(model_id)
        # Model supports CausalLM — tie encoder & decoder for efficiency
        tie_enc_dec = True
        decoder_path = model_id
    except ValueError:
        # Fall back to a separate lightweight decoder
        tie_enc_dec = False
        decoder_path = "bert-base-uncased"
        print(f"  (Using separate decoder '{decoder_path}' — "
              f"{model_id} has no CausalLM head)")

    train_loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=decoder_path,
        tie_encoder_decoder=tie_enc_dec,
    )

    tsdae_dir = output_dir / "tsdae-checkpoint"
    t0 = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5},
        show_progress_bar=True,
        output_path=str(tsdae_dir),
    )
    elapsed = time.time() - t0
    print(f"\nTSDAE completed in {elapsed / 60:.1f} min")
    print(f"Checkpoint saved to {tsdae_dir}")

    return str(tsdae_dir)


def run_mnrl(model_path: str, model_name: str, output_dir: Path,
             batch_size: int = 32, epochs: int = 3,
             lr: float = 2e-5, warmup_ratio: float = 0.1,
             eval_steps: int = 100):
    """Stage 2: MNRL supervised fine-tuning."""
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from sentence_transformers.similarity_functions import SimilarityFunction

    device = _detect_device()
    print(f"\n{'='*60}")
    print(f"  STAGE 2: MNRL — {model_path}")
    print(f"  Device: {device} | Batch: {batch_size} | Epochs: {epochs}")
    print(f"  LR: {lr} | Warmup: {warmup_ratio} | Eval every {eval_steps} steps")
    print(f"{'='*60}\n")

    model = SentenceTransformer(model_path, device=device)

    train_anchors, train_positives = _load_training_pairs("train")
    val_anchors, val_positives = _load_training_pairs("val")

    print(f"Train pairs: {len(train_anchors)}")
    print(f"Val pairs:   {len(val_anchors)}")

    train_dataset = Dataset.from_dict({
        "anchor": train_anchors,
        "positive": train_positives,
    })

    val_dataset = Dataset.from_dict({
        "anchor": val_anchors,
        "positive": val_positives,
    })

    loss = losses.MultipleNegativesRankingLoss(model)

    val_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=val_anchors,
        sentences2=val_positives,
        scores=[1.0] * len(val_anchors),
        main_similarity=SimilarityFunction.COSINE,
        name="val",
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "mnrl-checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        fp16=False,
        bf16=False,
        dataloader_drop_last=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=val_evaluator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nMNRL completed in {elapsed / 60:.1f} min")

    model.save(str(output_dir))
    print(f"Final model saved to {output_dir}")

    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage fine-tuning: TSDAE + MNRL")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="minilm",
                        help="Base model to fine-tune (default: minilm)")
    parser.add_argument("--tsdae-batch", type=int, default=16)
    parser.add_argument("--tsdae-epochs", type=int, default=1)
    parser.add_argument("--mnrl-batch", type=int, default=32)
    parser.add_argument("--mnrl-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--skip-tsdae", action="store_true",
                        help="Skip TSDAE stage, fine-tune from base model directly")
    parser.add_argument("--tsdae-only", action="store_true",
                        help="Run only TSDAE stage")
    args = parser.parse_args()

    model_id = MODELS[args.model]["id"]
    output_dir = MODELS_DIR / f"{args.model}-finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {model_id}")
    print(f"Output dir: {output_dir}")
    print(f"Device:     {_detect_device()}")

    if args.skip_tsdae:
        tsdae_ckpt = output_dir / "tsdae-checkpoint"
        if tsdae_ckpt.exists():
            stage1_path = str(tsdae_ckpt)
            print(f"\nSkipping TSDAE — using existing checkpoint: {tsdae_ckpt}")
        else:
            stage1_path = model_id
            print("\nSkipping TSDAE (--skip-tsdae), no checkpoint found")
    else:
        stage1_path = run_tsdae(
            args.model, output_dir,
            batch_size=args.tsdae_batch,
            epochs=args.tsdae_epochs,
        )

    if args.tsdae_only:
        print("\nStopping after TSDAE (--tsdae-only)")
        return

    run_mnrl(
        stage1_path, args.model, output_dir,
        batch_size=args.mnrl_batch,
        epochs=args.mnrl_epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
    )

    print(f"\nFine-tuning complete. Model at: {output_dir}")
    print(f"\nTo evaluate:")
    print(f"  python -m eval.loaders.jiraconfsd --large-scale --models {args.model}")


if __name__ == "__main__":
    main()
