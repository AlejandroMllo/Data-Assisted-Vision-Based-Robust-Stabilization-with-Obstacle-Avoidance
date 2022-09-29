import numpy as np

from joblib import dump, load
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR

from data.data_load import DataLoad


def train(data_loader, classifier_id, model_name=None):

    x, y = data_loader.get_data()
    # y = np.array([l[1] for l in y])

    print('Train Data: x.shape = {}, y.shape = {}'.format(x.shape, y.shape))
    if classifier_id == 'ridge':
        classifier = Ridge(alpha=1.0)
    elif classifier_id == 'mlp':
        classifier = MLPClassifier(
            solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
    elif classifier_id == 'linear_regression':
        classifier = LinearRegression()
    elif classifier_id == 'svr':
        classifier = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    classifier.fit(x, y)

    if isinstance(model_name, str):
        save_path = './saved_models/{}.joblib'.format(model_name)
        dump(classifier, save_path)

    return classifier


def test(model, data_loader):

    x, y = data_loader.get_data()
    # y = np.array([l[1] for l in y])
    score = model.score(x, y)

    print('Test Data: x.shape = {}, y.shape = {}'.format(x.shape, y.shape))
    print('Model Score = {}'.format(score))

    import matplotlib.pyplot as plt
    for i in [0, 43, 51, 4]:
        img = x[i]
        label = y[i]
        pred = model.predict(img.reshape(1, -1))

        img = img.reshape(img_shape)
        plt.imshow(img, cmap='gray')
        plt.title('Label = {} | Prediction = {}'.format(label, pred))
        plt.show()


def get_model(model_name):

    model_path = './saved_models/{}.joblib'.format(model_name)
    model = load(model_path)

    return model


if __name__ == '__main__':

    base_path = '/home/alejandro/Documents/Projects/Navigation/Linking_Perception_to_Control/data/generated_dataset/50x50/'
    model_name = 'ridge_regression_v0.1'

    img_shape = (50, 50)   # (300, 500)
    train_data_loader = DataLoad(base_path, 'train')
    val_data_loader = DataLoad(base_path, 'validation')

    _ = train(train_data_loader, 'ridge', model_name)
    model = get_model(model_name)
    test(model, train_data_loader)
    test(model, val_data_loader)
