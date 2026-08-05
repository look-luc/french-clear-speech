import sys
from pathlib import Path

import torch
from transformers import (
    AutoFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from .data import Data


def data_collate(batch, processor, feature_extractor):
    feature_list = [item["input_feature"] for item in batch]
    label_list = [item["labels"] for item in batch]

    padding_inputs = feature_extractor.pad(feature_list, return_tensors='pt')
    padded_labels = feature_extractor.pad(label_list, return_tensors='pt', padding_value=-100)

    return {
        "input_features": padding_inputs.input_features,
        "labels": padded_labels.input_ids
    }

class French_Speech_text:
    def __init__(
        self,
        model_id:str="bofenghuang/whisper-medium-french",
        level_tweak:float=0.0,
        device:str="cuda:0" if torch.cuda.is_available() else "mps:1" if  torch.backends.mps.is_available() else "cpu:2",
    ) -> None:
        self.device = torch.device(device)
        self.model_id = model_id

        self.feature_extractor, self.processor, self.model = self._setup()

        self.level_tweak = level_tweak

    def _setup(self):
        processor = WhisperProcessor.from_pretrained(self.model_id)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        return processor, feature_extractor, model
