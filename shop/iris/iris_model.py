import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder


class IrisModel:
    data_path = "C:/Users/MSJ/AIA/djangoProject/shop/iris/data"

    def __init__(self):
        self.cvs = pd.read_csv(f'{self.data_path}/Iris.csv')
        self.iris = datasets.load_iris()
        print(f'type {type(self.iris)}')
        self._x = self.iris.data
        self._y = self.iris.target


    def hook(self):
        self.spec()
        # self.create_model()


    def spec(self):
        print(self.iris['target_names'])

        '''
       Shape (150, 6)
       ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']
       '''

    def create_model(self):
        x = self._x
        y = self._y
        enc = OneHotEncoder()
        y_1hot = enc.fit_transform(y.reshape(-1, 1)).toarray()
        model = Sequential() #nn
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x, y_1hot, epochs=300, batch_size=10) #학습
        print('Model Training is completed')

        file_name = './save/iris_model.h5' #인코딩
        model.save(file_name)
        print(f'Model Saved in {file_name}')



iris_menu = ["Exit", #0
                "Hook",#1
            ]
iris_lambda = {
    "1" : lambda x: x.hook()
            }
if __name__ == '__main__':
    iris = IrisModel()
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

