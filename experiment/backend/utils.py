import io

import torchaudio

from python.transcription_model.model_experiment import French_Clear_Speech_Model


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
        audio_bytes = io.BytesIO(uploaded_file.read())
        audio_array, sample_rate = torchaudio.load(audio_bytes)
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
