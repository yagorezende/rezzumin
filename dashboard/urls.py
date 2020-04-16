from django.conf.urls import url
from .views import *

urlpatterns=[
    url(r'^$', index, name="index"),
    url(r'^result', result, name="result"),
    url(r'^process', process, name="process"),
]