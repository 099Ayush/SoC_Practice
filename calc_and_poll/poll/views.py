from django.shortcuts import render
from django.http import HttpResponse
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

options = [
    'Web and Coding Club',
    'Maths and Physics Club',
    'Electronics and Robotics Club',
    'Aeromodelling Club'
]

counts = {option: 0 for option in options}


def poll(request):
    return render(request, 'poll.html', {'options': options})


def poll_(request):
    counts[request.POST['club']] += 1
    return render(request, 'poll_.html')


def poll_admin(request):
    return render(request, 'poll_admin.html', {'counts': counts, 'options': options})