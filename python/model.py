import torch
from transformers import (
    AutoFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .data import get_dataloader


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

        self.processor, self.feature_extractor, self.model, self.data = self._setup()

        self.level_tweak = level_tweak

    def _setup(self):
        processor = WhisperProcessor.from_pretrained(self.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        data = get_dataloader(processor, feature_extractor, data_collate)

        return processor, feature_extractor, model, data
