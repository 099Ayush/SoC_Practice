from django.urls import path
from . import views
urlpatterns = [
    path('poll', views.poll),
    path('poll_', views.poll_),
    path('poll_admin', views.poll_admin)
]
