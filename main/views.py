
# Create your views here.
import requests
import json
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from . import prediction as rdf
import sys
import pandas as pd

# Create your views here.

def index(request):
    
    if request.method == 'POST':
        
        model = settings.MODEL_FILE
        val = request.POST.get('url')
        return render(request,"main/index.html",{"output":rdf.detect_flair(val,model)[0]})

    return render(request,"main/index.html")