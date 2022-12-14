from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf
from shop.iris.iris_service import IrisService


# Create your views here.

@api_view(['POST'])
@parser_classes([JSONParser])
def iris(request):
    iris = IrisService()
    try:
        user_info = request.data
        print("데이터 가져오기 ok")
        sepalLengthCm = tf.constant(float(user_info['SepalLengthCm']))
        sepalWidthCm = tf.constant(float(user_info['SepalWidthCm']))
        petalWidthCm = tf.constant(float(user_info['PetalLengthCm']))
        petalLengthCm = tf.constant(float(user_info['PetalWidthCm']))
        print(f'리액트에서 보낸 데이터 {user_info}')
        print(f'넘어온 꽃잎 길이 {petalLengthCm}')
        print(f'넘어온 꽃잎 폭 {petalWidthCm}')
        print(f'넘어온 꽃받침 길이 {sepalLengthCm}')
        print(f'넘어온 꽃받침 폭 {sepalWidthCm}')
        result = iris.service_model([sepalLengthCm, sepalWidthCm,
                                    petalWidthCm, petalLengthCm])
        print(f"result의 타입:{type(result)}")
        print(f"result :{result}")
        if result == 0:
            resp = 'setosa / 부채붓꽃'
        elif result == 1:
            resp = 'versicolor / 버시칼라'
        elif result == 2:
            resp = 'virginica / 버지니카'
        return JsonResponse({"붓꽃품종": resp})
    except:
        print("고치세요")

