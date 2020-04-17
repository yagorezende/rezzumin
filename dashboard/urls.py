from django.conf.urls import url
from .views import *

urlpatterns=[
    url(r'^$', index, name="index"),
    url(r'^result', result, name="result"),
    url(r'^process', process, name="process"),
    url(r'^process_text', process_text, name="process_text"),
    url(r'^get_status', get_status, name="get_status"),
]