from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from shop.iris.iris_model import IrisModel

# Create your views here.

@api_view(['GET'])
@parser_classes([JSONParser])
def iris(request):
    try:
        IrisModel().hook()
        print(f'Enter Stroke with {request}')
        print("\n********** 잘 됨 **********\n")
    except:
        print("고치세요")
    return JsonResponse({'Response Test ': 'SUCCEESS'})