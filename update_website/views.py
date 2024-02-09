from django.http import HttpResponse
from django.template import loader
from .retrival import get_top_N_images
from django.core.files.storage import FileSystemStorage
from PIL import Image
from django.conf import settings
import os

def index(request):    

    template = loader.get_template('searchview.html')
    rs_list = []
    
    if (request.method == "POST"):
        Is_query_text = True if request.POST['textSearch'] != "" else False
        Is_query_img = True if 'imageSearch' in request.FILES else False


       
        if(Is_query_text and not(Is_query_img)):
            query_text = request.POST['textSearch']
            rs = get_top_N_images(query_text, top_K=10, search_criterion="text")
            rs_list = rs[0].image_name.values

        if(Is_query_img and not(Is_query_text)):
            query_img = request.FILES['imageSearch']
            fss = FileSystemStorage()
            file = fss.save(query_img.name, query_img)

            
            query_image = Image.open(f"{settings.BASE_DIR}/{fss.url(file)}")
            os.remove(f"{settings.BASE_DIR}/{fss.url(file)}")

            rs = get_top_N_images(query_image, top_K=10, search_criterion="img")
            rs_list = rs[0].image_name.values
        precision = rs[1][0]
        recall = rs[1][1]
        f1_score = rs[1][2]
    context = {
        'animalresults': rs_list,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    return HttpResponse(template.render(context, request))
