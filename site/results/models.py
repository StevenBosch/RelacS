from __future__ import unicode_literals

from django.db import models

class Recording(models.Model):
    filename = models.CharField(max_length=255)
    length = models.IntegerField()
    stressful = models.IntegerField()
    relaxing = models.IntegerField()

    def __str__(self):
        return self.filename

class Sound(models.Model):
    recording = models.ForeignKey(Recording)
    start = models.IntegerField()
    end = models.IntegerField()
    stressful = models.IntegerField()
    relaxing = models.IntegerField()

    def __str__(self):
        return self.recording.filename + '-' + str(self.pk)

    class Meta:
        ordering = ('-stressful',)

class Category(models.Model):
    sound = models.ForeignKey(Sound)
    weight = models.FloatField()
    text = models.CharField(max_length=32)

    def __str__(self):
        return str(self.sound) + '-' + self.text

    class Meta:
        verbose_name_plural = 'categories'
