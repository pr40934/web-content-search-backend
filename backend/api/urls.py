from django.urls import path
from .views import test_search

urlpatterns = [
    path('search/', test_search),
]
