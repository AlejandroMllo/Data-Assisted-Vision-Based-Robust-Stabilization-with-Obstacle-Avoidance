"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
"""

"""
## Setup
"""
import matplotlib.pyplot as plt

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from data.data_load import DataLoad


def train_cnn(datasets):

    model = build_model()

    for ds in datasets:

        train_data_loader = DataLoad(base_path, ds)
        data = train_data_loader.get_data()

        print('Training with dataset: {}'.format(ds))
        print('Train data shape: {}'.format(data[0].shape))

        model = train(model, data)

    model.save(model_path)


def train(model, data):

    x_train, y_train = data

    model.compile(loss="mse", optimizer="adam", metrics=["mse", "mae"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    # model.save(model_path)
    return model


def build_model():

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(output_size)
        ]
    )

    return model


def test():

    data_split = 'test'
    val_data_loader = DataLoad(base_path, data_split)
    (x_test, y_test) = val_data_loader.get_data()

    model = keras.models.load_model(model_path)

    # for i in [4, 23, 43, 65, 200]:
    #     img = np.expand_dims(x_test[i], axis=0)
    #     label = y_test[i]
    #
    #     pred = model(img)
    #
    #     plt.imshow(img[0], cmap='gray')
    #     plt.title('Label = {} | Pred = {}'.format(label, pred[0]))
    #     plt.show()

    """
    ## Evaluate the trained model
    """
    score = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    print('Model metrics on', data_split, 'data:')
    desc = '\t{}: {}'
    for k, v in score.items():
        print(desc.format(k, v))
    print(desc.format('Support', len(y_test)))


if __name__ == '__main__':

    model_path = 'saved_models/cnn_100kTrain'    # 'cnn_100kTrain'   'cnn_100kTrain_100%_OcclusionRate'

    # Model / data parameters
    output_size = 2
    input_shape = (60, 100, 3)
    batch_size = 128
    epochs = 5

    model = build_model()
    print(model.summary())

    # the data, split between train and test sets
    base_path = '/home/alejandro/Documents/Projects/Navigation/Linking_Perception_to_Control/data/' + \
                'generated_dataset/100x60'     # _100%_OcclusionRate'
    # train_datasets = ['train' + str(i) for i in range(5)]

    # Run
    # train_cnn(train_datasets)
    test()
