import pandas as pd
import tensorflow as tf
from keras.saving.save import load_model
from sklearn import datasets
import numpy as np

class IrisService:

    def __init__(self):
        global model, graph, target_names
        model = load_model('./save/iris_model.h5')
        graph = tf.get_default_graph()
        target_names = datasets.load_iris().target_names

    def hook(self):
        features = [

        ]

        features = np.reshape(features, (1, 4))


    def service_model(self):
        pass




iris_menu = ["Exit", #0
                "Hook",#1
            ]
iris_lambda = {
    "1" : lambda x: x.hook()
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
