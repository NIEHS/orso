from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse

from . import models


def index(request):
    return HttpResponse('Hello, world.')


def dataset(request, pk):
    dataset = get_object_or_404(models.Dataset, pk=pk)
    return render(request, 'dataset.html', {'dataset': dataset})
