from django.shortcuts import render
from django.http import HttpResponse

def calc(request):
    return render(request, 'calc.html')

def calculate(request):
    exp = request.POST['exp']
    exp = exp.replace('^', '**').replace('|', '//')
    return render(request, 'result.html', {'result': eval(exp)})