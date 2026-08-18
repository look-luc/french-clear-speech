import sys
import tempfile
from pathlib import Path

import scipy

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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            audio_array, sample_rate = scipy.io.wavfile.read(temp_file.name)
        return audio_array

    def execute(self, file, cutoff_freq=None, snr_db=None, temp=1.0):
        audio_array = self.decode_audio(file)

        self.transcription, self.confidence = self.model.transcribe(
            audio_array=audio_array.numpy(),
            sampling_rate=16000,
            cutoff_freq=cutoff_freq,
            snr_db=snr_db,
            temp=temp,
        )
        return self.transcription
