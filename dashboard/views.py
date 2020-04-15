from django.shortcuts import render, redirect


def index(request):
    return render(request, 'dashboard/index.html')

def process(request):
    return redirect('/result', result="something")

def result(request):
    print(request.result)
    return render(request)
