from src.estimator.Estimator import Estimator
import sklearn.linear_model as skll
import numpy as np


class LinearRegressionEstimator(Estimator):

    def __init__(self):
        self.estimator = skll.LinearRegression()

    def __call__(self, state, action):
        ti = np.array([state[0], state[1], action]).reshape(1, -1)
        return self.estimator.predict(ti)[0]

    def train(self, train_in, train_out):
        ti = []
        for sample in train_in:
            ti.append([sample[0][0], sample[0][1], sample[1]])
        ti = np.array(ti)
        self.estimator.fit(ti, train_out)
