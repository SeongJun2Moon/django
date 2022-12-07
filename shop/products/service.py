from tensorflow import keras

class MyFashion(keras.Model):
    def __init__(self):
        pass

    def exec(self):
        (train_input, train_target), (train_input, train_target) = keras.datasets.fashion_mnist.load_data()