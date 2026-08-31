from django.db import models


class audio_experiment_record(models.Model):
    primary_key = models.AutoField(primary_key=True, db_column="primary_key")
    subject = models.CharField(max_length=100)
    stimulus = models.CharField(max_length=128)
    trial_index = models.CharField(max_length=30, null=True, blank=True)
    response = models.TextField()
    custom_tag = models.CharField(max_length=100, default="clear_speech")

    class Meta:
        db_table = "data_response"
