from django.contrib import admin
from .models import Data
# Register your models here.

@admin.register(Data)
class DataAdmin(admin.ModelAdmin):
    list_display = ['timestamp','p13','edge','exed']
    list_filter = ['timestamp','p13','edge','exed']
