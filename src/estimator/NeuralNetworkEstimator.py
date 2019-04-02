from src.estimator.Estimator import Estimator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler


class NeuralNetworkEstimator(Estimator):

    def __init__(self):

        opt = SGD(lr=0.01, momentum=0.9)

        self.estimator = Sequential()
        self.estimator.add(Dense(3, input_dim=3, activation='relu'))
        self.estimator.add(Dense(3, activation='relu'))
        self.estimator.add(Dense(3, activation='relu'))
        self.estimator.add(Dense(1, activation='linear'))
        self.estimator.compile(loss='mean_squared_error', optimizer="adam", metrics=['mse'])

        self.scaler= MinMaxScaler()

    def __call__(self, state, action):
        ti = np.array([state[0], state[1], action]).reshape(1, -1)
        self.scaler.transform(ti)
        return self.estimator.predict(ti)[0]

    def train(self, train_in, train_out):
        ti = []
        for sample in train_in:
            ti.append([sample[0][0], sample[0][1], sample[1]])
        ti = np.array(ti)
        self.scaler.fit_transform(ti)
        self.estimator.fit(ti, train_out, epochs=50)
