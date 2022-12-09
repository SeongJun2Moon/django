from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from movies.movies.servieces import DcGan

# Create your views here.

@api_view(['GET'])
@parser_classes([JSONParser])
def fake_faces(request):
    DcGan().show_image()
    print(f'Enter Show Faces with {request}')
    return JsonResponse({'Response Test ': 'SUCCEESS'})