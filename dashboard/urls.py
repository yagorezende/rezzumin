from django.conf.urls import url
from .views import *

urlpatterns=[
    url('', index, name="index"),
    url('/result', result, name="result"),
    url('/process', process, name="process"),
]