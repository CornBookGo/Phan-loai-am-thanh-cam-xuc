from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

classification_grid_parameters = {
    SVC():  {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma' : [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    KNeighborsClassifier(): {
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    MLPClassifier():    {
        'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500]
    },
}

regression_grid_parameters = {
    KNeighborsRegressor(): {
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    MLPRegressor():    {
        'hidden_layer_sizes': [(200,), (200, 200), (300,), (400,)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [64, 128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300, 400, 500, 600, 700]
    },
}