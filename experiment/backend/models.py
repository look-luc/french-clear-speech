from django.db import models


class audio_experiment_record(models.Model):
    audio_file = models.FileField(upload_to="jspsych_audio/")
    transcription = models.FileField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
