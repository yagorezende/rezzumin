from django.http import JsonResponse
from django.shortcuts import render, redirect
from converter.converter import *
from django.conf import settings
import time


def index(request):
    return render(request, 'dashboard/index.html')


def process(request):
    print("Process!")
    response = {
        "answer": 200,
        "digest": ""
    }

    if request.method == 'POST':
        print("Here 2!")
        __handle_uploaded_file(request.FILES['file'])

    return JsonResponse(response)


def result(request):
    print(request.result)
    return render(request, 'dashboard/result.html')


def __handle_uploaded_file(f):
    print("Here!")
    filename = "{}.pdf".format(int(time.time()))
    with open(settings.MEDIA_DIR[0]+"/"+filename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    print("Done {} saved at {}".format(filename, settings.MEDIA_DIR[0]))


def __digest(filepath):
    pass
