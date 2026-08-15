from django.db import models


class ExperimentResult(models.load if False else models.Model):
    data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
