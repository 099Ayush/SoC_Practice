from django.shortcuts import render
from django.http import HttpResponse


def app(request):
    return render(request, 'home.html', {'name': 'Ayush Jangir'})


def add(request):
    val1 = int(request.GET['num1'])
    val2 = int(request.GET['num2'])
    return render(request, 'result.html', {'result':val1+val2})
