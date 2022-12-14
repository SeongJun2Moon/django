import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.saving.save import load_model
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

class IrisService:

    def __init__(self):
        global model, graph, target_names
        model = load_model('C:/Users/MSJ/AIA/djangoProject/shop/iris/save/iris_model.h5')
        print(type(model))
        target_names = datasets.load_iris().target_names

    def service_model(self, features):
        #features = []
        features = np.reshape(features, (1, 4)) # 리스트를 행렬로 바꿈
        y_prob = model.predict(features, verbose=0) # 모델에 행렬 넣어서 확률 출력
        print(f"y-prob타입:{type(y_prob)}")
        predicted = y_prob.argmax(axis=-1) # axis=-1=>가장 낮은 차원 argmax=>가장 높은 값
        return predicted




iris_menu = ["Exit", #0
                "Hook",#1
            ]
iris_lambda = {
    "1" : lambda x: x.service_model()
            }
if __name__ == '__main__':
    iris = IrisService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")