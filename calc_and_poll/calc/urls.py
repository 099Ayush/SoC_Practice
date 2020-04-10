from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('calc', views.calc),
    path('calculate', views.calculate)
]
