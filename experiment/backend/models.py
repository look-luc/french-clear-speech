import numpy as np
from django.db import models

from python.transcription_model.model_experiment import French_Clear_Speech_Model


class ExperimentResult(models.load if False else models.Model):
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

class audio_processor:
    def __init__(self, sample_rate:int=16000, duration:float=5, channels:int=1) -> None:
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels

        self.samples = self.sample_rate * self.duration

    def validate_django_file(self, file):
        allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/flac"]
        max_size_bytes = 50 * 1024 * 1024

        if file.size > max_size_bytes:
            raise ValueError("File size exceeds limit")
        if file.content_type not in allowed_types:
            raise ValueError("Unsupported audio format")
        return True
        self.transcription, self.confidence = self.model.transcribe(
            audio_array=audio_array.numpy(),
            sampling_rate=16000,
            cutoff_freq=cutoff_freq,
            snr_db=snr_db,
            temp=temp,
        )
        return self.transcription


class audio_job(models.Model):
    audio_file = models.FileField(upload_to="audio_uploads/")
    status = models.CharField(
        max_length=20,
        choices=(
            ("PENDING", "Pending"),
            ("PROCESSING", "Processing"),
            ("COMPLETED", "Completed"),
            ("FAILED", "Failed"),
        ),
        default="PENDING"
    )
    results = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
