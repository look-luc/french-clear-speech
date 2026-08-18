import os
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from model.transcription_model.model_experiment import French_Clear_Speech_Model


class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = French_Clear_Speech_Model()
        return cls._instance

class transcription:
    def __init__(self) -> None:
        self.model = ModelSingleton.get_instance()

    def decode_audio(self, uploaded_file):
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as in_file, \
                tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:

            in_path = in_file.name
            out_path = out_file.name

            for chunk in uploaded_file.chunks():
                in_file.write(chunk)
            in_file.flush()

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", out_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            audio_array, sample_rate = sf.read(out_path, dtype="float32")

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            audio_tensor = torch.from_numpy(audio_array)

            return audio_tensor

        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)

    def execute(self, file, cutoff_freq=None, snr_db=None, temp=1.0):
        audio_tensor = self.decode_audio(file)

        self.transcription, self.confidence = self.model.transcribe(
            audio_array=audio_tensor,
            sampling_rate=16000,
            cutoff_freq=cutoff_freq,
            snr_db=snr_db,
            temp=temp,
        )
        return self.transcription, self.confidence
