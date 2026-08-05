import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from .data import get_data


class French_Speech_text:
    def __init__(
        self,
        model_id:str="bofenghuang/whisper-medium-french",
        level_tweak:float=0.0,
        device:str="cuda:0" if torch.cuda.is_available() else "mps:1" if  torch.backends.mps.is_available() else "cpu:3",
    ) -> None:
        self.device = torch.device(device)
        self.model_id = model_id

        self.model, self.processor = self._setup()

        self.level_tweak = level_tweak

    def _setup(self):
        processor = WhisperProcessor.from_pretrained(self.model_id)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        return processor, model
