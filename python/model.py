import evaluate
import numpy as np
import torch
from torchmetrics.functional.text import bleu_score
from torchmetrics.text import EditDistance
from transformers import (
    AutoFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .data import get_train_test_datasets

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")
f1_metric = EditDistance()

def data_collate(batch, processor, feature_extractor):
    feature_list = [{"input_features": item["input_features"]} for item in batch]
    label_list = [{"input_ids": item["labels"]} for item in batch]

    padded_inputs = feature_extractor.pad(feature_list, return_tensors="pt")
    padded_labels = processor.tokenizer.pad(label_list, return_tensors="pt", padding_value=-100)

    return {
        "input_features": padded_inputs.input_features,
        "labels": padded_labels.input_ids,
    }

class French_Speech_text:
    def __init__(
        self,
        model_id:str="bofenghuang/whisper-medium-french",
        level_tweak:float=0.0,
        device:str=""
    ) -> None:
        if device == "" or device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if  torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_id = model_id

        self.processor, self.feature_extractor, self.model, self.train_split, self.test_split = self._setup()
        self.model

        self.level_tweak = level_tweak

    def _setup(self):
        processor = WhisperProcessor.from_pretrained(self.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        train_dataset, test_dataset = get_train_test_datasets(processor, feature_extractor)

        return processor, feature_extractor, model, train_dataset, test_dataset

    def _compute_metrics(self, eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        pad_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        clean_label_ids = np.where(label_ids != -100, label_ids, pad_id)
        clean_pred_ids = np.where(label_ids != -100, pred_ids, pad_id)

        decoded_preds = self.processor.tokenizer.batch_decode(
            clean_pred_ids, skip_special_tokens=True
        )
        decoded_labels = self.processor.tokenizer.batch_decode(
            clean_label_ids, skip_special_tokens=True
        )

        decoded_preds = [
            pred.strip() if pred.strip() else " " for pred in decoded_preds
        ]
        decoded_labels = [
            label.strip() if label.strip() else " " for label in decoded_labels
        ]

        cer_score = cer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        wer_score = wer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        bleu_targets = [[label] for label in decoded_labels]

        f1_metric.update(decoded_preds, decoded_labels)
        f1_score = f1_metric.compute()
        f1_metric.reset()

        try:
            bleu_score_val = bleu_score(decoded_preds, bleu_targets).item()
        except Exception:
            bleu_score_val = 0.0

        return {"F1": f1_score, "CER": cer_score, "WER": wer_score, "BLEU": bleu_score_val}

    def train(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            dataloader_persistent_workers=True,
            num_train_epochs=1,
            learning_rate=2e-5,
            max_steps=2500,
            eval_strategy="steps",
            eval_steps=500,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            warmup_steps=125,
            lr_scheduler_type="cosine",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_split,
            eval_dataset=self.test_split,
            data_collator=lambda batch: data_collate(batch, self.processor, self.feature_extractor),
            compute_metrics=self._compute_metrics,
        )

        return trainer.train()
