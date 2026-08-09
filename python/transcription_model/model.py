import sys
from functools import partial
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

python_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]

if str(python_dir) not in sys.path:
    sys.path.append(str(python_dir))

from python.get_data.get_data import get_data

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def data_collate(batch, processor, feature_extractor):
    feature_list = [{"input_features": item["input_features"]} for item in batch]
    label_list = [{"input_ids": item["labels"]} for item in batch]

    padded_inputs = feature_extractor.pad(feature_list, return_tensors="pt")
    padded_labels = processor.tokenizer.pad(label_list, return_tensors="pt")

    labels = padded_labels["input_ids"].masked_fill(
        padded_labels["input_ids"] == processor.tokenizer.pad_token_id, -100
    )

    return {
        "input_features": padded_inputs.input_features,
        "labels": labels,
    }


class French_Speech_text:
    def __init__(
        self,
        model_id: str = "bofenghuang/whisper-medium-french",
        level_tweak: float = 0.0,
        device: str = "",
        freeze_encoder: bool = True,
    ) -> None:
        if device == "" or device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model_id = model_id
        self.processor, self.feature_extractor, self.model, self.train_split, self.test_split = self._setup()
        self.level_tweak = level_tweak

        # Freeze acoustic encoder to speed up fine-tuning and save memory
        if freeze_encoder:
            self.model.freeze_encoder()

    def _setup(self):
        processor = WhisperProcessor.from_pretrained(self.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id, use_safetensors=True)

        train_dataset, test_dataset = get_data(
            processor,
            feature_extractor
        )

        return processor, feature_extractor, model, train_dataset, test_dataset

    def _compute_metrics(self, eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pad_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        clean_label_ids = np.where(label_ids != -100, label_ids, pad_id)

        decoded_preds = self.processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        decoded_labels = self.processor.tokenizer.batch_decode(
            clean_label_ids, skip_special_tokens=True
        )

        decoded_preds = [pred.strip() if pred.strip() else " " for pred in decoded_preds]
        decoded_labels = [label.strip() if label.strip() else " " for label in decoded_labels]

        cer_score = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        wer_score = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {"CER": cer_score, "WER": wer_score}

    def train(self, output_dir: str = "./whisper-french-final"):
        is_cuda = torch.cuda.is_available()
        use_bf16 = is_cuda and torch.cuda.is_bf16_supported()

        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            dataloader_pin_memory=is_cuda,
            dataloader_prefetch_factor=2 if is_cuda else None,
            gradient_checkpointing=True,
            dataloader_num_workers=4 if is_cuda else 1,
            dataloader_persistent_workers=is_cuda,
            num_train_epochs=1,
            learning_rate=3e-5,
            max_steps=400,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_CER",
            greater_is_better=False,
            logging_steps=15,
            predict_with_generate=True,
            generation_max_length=200,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            bf16=use_bf16,
            fp16=(is_cuda and not use_bf16),
            use_cpu=not is_cuda,
        )

        picklable_collator = partial(
            data_collate,
            processor=self.processor,
            feature_extractor=self.feature_extractor
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_split,
            eval_dataset=self.test_split,
            data_collator=picklable_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        self.feature_extractor.save_pretrained(output_dir)

        return train_result

    def transcribe(self):
        pass
