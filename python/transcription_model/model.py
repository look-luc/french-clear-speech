import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from torchmetrics.text import EditDistance
from transformers import (
    AutoFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

python_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]

if str(python_dir) not in sys.path:
    sys.path.append(str(python_dir))

from python.get_data.data import get_train_test_datasets

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")
f1_metric = EditDistance()

def data_collate(batch, processor, feature_extractor):
    feature_list = [{"input_features": item["input_features"]} for item in batch]
    label_list = [{"input_ids": item["labels"]} for item in batch]

    padded_inputs = feature_extractor.pad(feature_list, return_tensors="pt")
    padded_labels = processor.tokenizer.pad(label_list, return_tensors="pt", padding_side=-100)

    return {
        "input_features": padded_inputs.input_features,
        "labels": padded_labels.input_ids,
    }

class French_Speech_text:
    def __init__(
        self,
        model_id: str = "bofenghuang/whisper-medium-french",
        level_tweak: float = 0.0,
        device: str = ""
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

    def _setup(self):
        processor = WhisperProcessor.from_pretrained(self.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        praat_data_dir = root_dir / "praat" / "data"
        train_dataset, test_dataset = get_train_test_datasets(
            processor,
            feature_extractor,
            audio_dir=praat_data_dir,
            txt_dir=praat_data_dir
        )

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

        decoded_preds = [pred.strip() if pred.strip() else " " for pred in decoded_preds]
        decoded_labels = [label.strip() if label.strip() else " " for label in decoded_labels]

        cer_score = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        wer_score = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        f1_metric.update(decoded_preds, decoded_labels)
        f1_score = f1_metric.compute()
        f1_metric.reset()

        return {"F1": f1_score, "CER": cer_score, "WER": wer_score}

    def train(self, output_dir: str = "./whisper-french-final"):
        is_cuda = torch.cuda.is_available()
        use_bf16 = is_cuda and torch.cuda.is_bf16_supported()

        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            dataloader_pin_memory=is_cuda,
            dataloader_prefetch_factor=2 if is_cuda else None,
            gradient_checkpointing=True,
            dataloader_num_workers=4 if is_cuda else 0,
            dataloader_persistent_workers=is_cuda,
            num_train_epochs=1,
            learning_rate=2e-5,
            max_steps=2500,
            eval_strategy="steps",
            eval_steps=500,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            warmup_steps=125,
            lr_scheduler_type="cosine",
            bf16=use_bf16,
            fp16=(is_cuda and not use_bf16),
            use_cpu=not is_cuda,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_split,
            eval_dataset=self.test_split,
            data_collator=lambda batch: data_collate(batch, self.processor, self.feature_extractor),
            compute_metrics=self._compute_metrics,
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        self.feature_extractor.save_pretrained(output_dir)

        return train_result

    def transcribe(self):
        # model.generate(
        #     input_features,
        #     language="fr",
        #     task="transcribe",

        #     # 1. Simulate "Mishearing" via High Temperature
        #     do_sample=True,
        #     temperature=0.8,         # Forces the model to guess phonetically similar but wrong words

        #     # 2. Simulate "Missing Context" by limiting vision
        #     num_beams=1,             # No look-ahead context; forces rigid word-by-word guessing

        #     # 3. Simulate "Giving Up" early on weak signals
        #     logprob_threshold=-0.4,  # If a word isn't crystal clear, Whisper stops or drops it

        #     # 4. Prevent repetitive guessing
        #     no_repeat_ngram_size=3   # Stops the model from looping if it gets confused by a sound
        # )
        pass
