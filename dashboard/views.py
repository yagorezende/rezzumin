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
        "result": 2
    }
    # if request.method == 'POST':
    #     print("Here 2!")
    #     file_request = __handle_uploaded_file(request.FILES['file'])
    #     if file_request.isSaved:
    #         file_request.saveAbs(__digest(file_request))
    #         response["result"] = file_request.id
    return JsonResponse(response)


def result(request):
    print(request.GET.get("id"))
    return render(request, 'dashboard/result.html')


def __handle_uploaded_file(f) -> ConvertRequest:
    print("Here!")
    # TODO validation (.pdf) 20 pages only too
    id = "".join([chr(int(65+random.random()*25)) for i in range(0,10)])
    fileRequest = ConvertRequest(id, settings.MEDIA_DIR[0])
    fileRequest.pdfFile = f
    fileRequest.isSaved = fileRequest.savePDFFile()
    return fileRequest



def __digest(file_request : ConvertRequest):
    file_request.fullText = pdf_txt_converter(file_request)
    file_request.cleanupText()
    text_reading(file_request)
    names_dict = get_name_dict(file_request)
    preprocessing(names_dict, file_request.path)
    position_weighted_metric(file_request.path)
    tfidf_metric(file_request.path)
    tfidf_cossine_euclidean(file_request.path)
    # text_rank_method()
    brush_path(file_request.path)
    output = k_medoids_method(file_request.path)
    # rouge_evaluation(file_request, output)
    return output
