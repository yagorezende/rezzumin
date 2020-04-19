# coding: UTF-8

from django.http import JsonResponse
from django.shortcuts import render, redirect
from converter.convert_request import ConvertRequest
from converter.converter import *
from django.conf import settings
import time
import random


def index(request):
    return render(request, 'dashboard/index.html')

def process(request):
    print("Process!")
    response = {
        "answer": 200,
        "result": None
    }
    if request.method == 'POST':
        rid = request.POST.get("rid")
        file_request = __handle_uploaded_file(request.FILES['file'], rid)
        file_request.portion = int(request.POST.get("percent"))
        file_request.saveStatus()
        if file_request.isSaved:
            file_request.saveAbs(__digest(file_request))
            response["result"] = file_request.id
    return JsonResponse(response)

def process_text(request):
    print("Process Text!")
    response = {
        "answer": 200,
        "result": None
    }

    if request.method == 'POST':
        rid = request.POST.get("rid")
        file_request = ConvertRequest(rid, settings.MEDIA_DIR[0])
        file_request.portion = int(request.POST.get("percent"))
        file_request.body = request.POST.get("text")
        file_request.saveStatus()
        file_request.saveAbs(__digest(file_request))
        response["result"] = file_request.id

    return JsonResponse(response)


def result(request):
    id = request.GET.get("id")
    file_request = ConvertRequest(id, settings.MEDIA_DIR[0])
    context = {"abstract": None}

    with open(file_request.getAbsPath(), 'r', encoding='utf8') as abs_file:
        context["abstract"] = abs_file.read()

    return render(request, 'dashboard/result.html', context)


def __handle_uploaded_file(f, rid) -> ConvertRequest:
    print("Here!")
    # TODO validation (.pdf) 20 pages only too
    # id = "".join([chr(int(65+random.random()*25)) for i in range(0, 10)])
    fileRequest = ConvertRequest(rid, settings.MEDIA_DIR[0])
    fileRequest.pdfFile = f
    fileRequest.isSaved = fileRequest.savePDFFile()
    fileRequest.isPdf = True
    return fileRequest


def get_status(request):
    id = request.GET.get("id")
    file_request = ConvertRequest(id, settings.MEDIA_DIR[0])

    response = {"answer": 200, "status": file_request.loadStatus()}

    return JsonResponse(response)


def __digest(file_request: ConvertRequest):
    if file_request.isPdf:
        file_request.fullText = pdf_txt_converter(file_request)  #+10%
        file_request.status = 30

        file_request.cleanupText()  #+10%
        file_request.incrementStatus(10)
    else:
        file_request.status = 30

    text_reading(file_request)  #+10%
    file_request.incrementStatus(10)

    names_dict = get_name_dict(file_request.path)  #+10%
    file_request.incrementStatus(10)

    preprocessing(names_dict, file_request.path)  #+10%
    file_request.incrementStatus(10)

    position_weighted_metric(file_request.path)  #+10%
    file_request.incrementStatus(10)

    graph_methodology(file_request.path)  #+10%
    file_request.incrementStatus(10)

    # tfidf_cossine_euclidean(file_request.path)  #+10%
    # file_request.incrementStatus(10)
    #
    # brush_path(file_request.path)  #+10%
    # file_request.incrementStatus(10)

    output = k_medoids_method(file_request)  #+10%
    file_request.incrementStatus(10)
    return output
