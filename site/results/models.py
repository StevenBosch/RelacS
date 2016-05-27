from __future__ import unicode_literals

from django.db import models

class Recording(models.Model):
    filename = models.CharField(max_length=255)
    length = models.IntegerField


class Sound(models.Model):
    recording = models.ForeignKey(Recording)
    start = models.IntegerField()
    end = models.IntegerField()
    stressful = models.FloatField()
    relaxing = models.FloatField()

class Category(models.Model):
    recording = models.ForeignKey(Sound)
    weight = models.FloatField()
