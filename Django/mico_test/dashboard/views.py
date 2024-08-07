from django.shortcuts import render
from django.http import JsonResponse
from .models import Data
import json

def index(request):
    return render(request, 'index.html')


def get_data(request):
    print('get_data 함수 실행')
    data_type = request.GET.get('data_type', 'p13')
    data = Data.objects.all().order_by('timestamp')

    timestamps = [entry.timestamp.strftime("%Y-%m-%d %H:%M:%S") for entry in data]
    values = [getattr(entry, data_type) for entry in data]

    response_data = {
        'timestamps': timestamps,
        'values': values,
    }
    
    print(response_data)
    return JsonResponse(response_data)