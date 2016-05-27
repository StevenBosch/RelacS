from django.contrib import admin
from results import models

class RecordingAdmin(admin.ModelAdmin):
    pass
admin.site.register(models.Recording, RecordingAdmin)

class SoundAdmin(admin.ModelAdmin):
    pass
admin.site.register(models.Sound, SoundAdmin)

class CategoryAdmin(admin.ModelAdmin):
    pass
admin.site.register(models.Category, CategoryAdmin)
