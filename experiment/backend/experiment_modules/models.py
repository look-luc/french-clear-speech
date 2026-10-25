from django.db import models


class Audio(models.Model):
    primary_key = models.AutoField(primary_key=True, db_column="primary_key")
    audio_str = models.CharField(max_length=128)

    class Meta:
        db_table = "audio"

class DataResponse(models.Model):
    primary_key = models.AutoField(primary_key=True, db_column="primary_key")
    subject = models.CharField(max_length=100)
    age = models.SmallIntegerField()
    gender = models.CharField(max_length=32)
    trial_index = models.CharField(max_length=30, null=True, blank=True)
    response = models.TextField()
    confidence = models.FloatField()
    custom_tag = models.CharField(max_length=100, default="clear_speech")

    audio_files = models.ManyToManyField(Audio, through="DataResponseAudio")

    class Meta:
        db_table = "data_response"

class DataResponseAudio(models.Model):
    id = models.AutoField(primary_key=True, db_column="id")
    data_response = models.ForeignKey(
        DataResponse,
        on_delete=models.CASCADE
    )
    audio = models.ForeignKey(
        Audio,
        on_delete=models.CASCADE
    )

    class Meta:
        db_table = "data_response_audio"
