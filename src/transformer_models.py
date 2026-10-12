"""
Multilingual Transformer module for Aspect-Based Sentiment Analysis.
Supports lazy loading and fine-tuning for mBERT, XLM-RoBERTa, and MuRIL.
Automatically detects CUDA, Apple Silicon (MPS), or CPU.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Mapping of supported models to Hugging Face model hub identifiers
SUPPORTED_TRANSFORMERS: Dict[str, str] = {
    "mBERT": "bert-base-multilingual-cased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "MuRIL": "google/muril-base-cased",
}

LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}


def get_optimal_device() -> torch.device:
    """Detects CUDA, Apple Silicon (MPS), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ABSADataset(Dataset):
    """PyTorch Dataset for Aspect-conditioned Review sequences."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer: Any, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@dataclass
class TrainingProgress:
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    message: str = ""


class MultilingualABSAModel:
    """
    Interface for mBERT, XLM-RoBERTa, and MuRIL models.
    Does not download weights on startup; loads on demand only.
    """

    def __init__(
        self,
        model_name: str = "mBERT",
        device: Optional[torch.device] = None,
        max_seq_len: int = 128,
    ):
        if model_name not in SUPPORTED_TRANSFORMERS:
            raise ValueError(f"Model '{model_name}' not supported. Options: {list(SUPPORTED_TRANSFORMERS.keys())}")

        self.model_name = model_name
        self.hf_checkpoint = SUPPORTED_TRANSFORMERS[model_name]
        self.device = device or get_optimal_device()
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.is_trained = False

    def load_model(self) -> None:
        """
        Lazily downloads/initializes tokenizer and classification model from Hugging Face.
        """
        if self.is_loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.hf_checkpoint,
                num_labels=3,
                id2label=ID_TO_LABEL,
                label2id=LABEL_TO_ID,
            )
            self.model.to(self.device)
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{self.hf_checkpoint}'. "
                f"Check internet connection and Hugging Face Hub availability. Details: {e}"
            )

    def prepare_input_text(self, aspect: str, review: str, opinion: Optional[str] = None) -> str:
        """Formats input sequence with explicit aspect and opinion focus."""
        asp = str(aspect or "").strip()
        op = str(opinion or "").strip()
        rev = str(review or "").strip()
        if op:
            return f"Aspect: {asp} | Opinion: {op} | Context: {rev}"
        return f"Aspect: {asp} | Context: {rev}"

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        save_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> Dict[str, Any]:
        """
        Fine-tunes the transformer model on the given dataset.
        """
        self.load_model()
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        # Prepare inputs
        train_texts = [
            self.prepare_input_text(row["aspect_term"], row["review_text"], row.get("opinion_term"))
            for _, row in train_df.iterrows()
        ]
        train_labels = [LABEL_TO_ID.get(row["sentiment"], 1) for _, row in train_df.iterrows()]

        train_ds = ABSADataset(train_texts, train_labels, self.tokenizer, self.max_seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_df is not None and len(val_df) > 0:
            val_texts = [
                self.prepare_input_text(row["aspect_term"], row["review_text"], row.get("opinion_term"))
                for _, row in val_df.iterrows()
            ]
            val_labels = [LABEL_TO_ID.get(row["sentiment"], 1) for _, row in val_df.iterrows()]
            val_ds = ABSADataset(val_texts, val_labels, self.tokenizer, self.max_seq_len)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps,
        )

        history = []
        best_val_f1 = -1.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**kwargs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_loader))

            # Validation
            avg_val_loss = 0.0
            val_f1 = 0.0
            if val_loader:
                self.model.eval()
                val_preds = []
                val_targets = []
                total_val_loss = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)

                        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                        if "token_type_ids" in batch:
                            kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                        outputs = self.model(**kwargs)
                        total_val_loss += outputs.loss.item()
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                        val_preds.extend(preds)
                        val_targets.extend(labels.cpu().numpy().tolist())

                avg_val_loss = total_val_loss / max(1, len(val_loader))
                from sklearn.metrics import f1_score
                val_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))

            epoch_stat = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "val_macro_f1": round(val_f1, 4),
            }
            history.append(epoch_stat)

            if progress_callback:
                progress_callback(TrainingProgress(
                    epoch=epoch,
                    total_epochs=epochs,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    val_macro_f1=val_f1,
                    message=f"Epoch {epoch}/{epochs} complete.",
                ))

            if save_dir and val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                self.save_checkpoint(save_dir)

        self.is_trained = True
        return {"history": history, "best_val_f1": best_val_f1}

    def predict(self, aspect: str, review: str, opinion: Optional[str] = None) -> Tuple[str, float]:
        """Predicts sentiment for a single aspect-review pair."""
        self.load_model()
        self.model.eval()

        text = self.prepare_input_text(aspect, review, opinion)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            confidence = float(probs[pred_id])

        return ID_TO_LABEL[pred_id], round(confidence, 2)

    def predict_batch(self, df: pd.DataFrame, batch_size: int = 16) -> Tuple[List[str], List[float]]:
        """Batch prediction over DataFrame."""
        self.load_model()
        self.model.eval()

        texts = [
            self.prepare_input_text(row["aspect_term"], row["review_text"], row.get("opinion_term"))
            for _, row in df.iterrows()
        ]
        ds = ABSADataset(texts, None, self.tokenizer, self.max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_preds = []
        all_confs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in batch:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**kwargs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                pred_ids = np.argmax(probs, axis=1)

                for pid, p_row in zip(pred_ids, probs):
                    all_preds.append(ID_TO_LABEL[int(pid)])
                    all_confs.append(round(float(p_row[pid]), 2))

        return all_preds, all_confs

    def save_checkpoint(self, output_dir: str) -> None:
        """Saves model weights, tokenizer, and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        if self.model and self.tokenizer:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            config_meta = {
                "model_name": self.model_name,
                "hf_checkpoint": self.hf_checkpoint,
                "max_seq_len": self.max_seq_len,
            }
            with open(os.path.join(output_dir, "absa_config.json"), "w") as f:
                json.dump(config_meta, f, indent=2)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Loads fine-tuned weights from checkpoint."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.is_loaded = True
        self.is_trained = True
