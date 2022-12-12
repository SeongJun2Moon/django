from django.urls import re_path as url
from stroke import views

urlpatterns = [
    url(r'stroke', views.stroke)
]

