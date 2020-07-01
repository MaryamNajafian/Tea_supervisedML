"""
Naive Bayes Classifier

Naive means all input features are independent
    P(X|C) = mult(p(X_i|C))

Non-Naive means we just can't equate the above (no independence)
    P(X|C) = HMMs

"""

import pandas as pd
import numpy as np
from future.utils import iteritems

from util import get_data
from scipy.stats import norm # single dimension gaussian
from scipy.stats import multivariate_normal as mvn
from datetime import datetime

def get_data(limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv('large_files/train.csv')
    # data = df.as_matrix() # turn data into numpy ndarray
    data = df.values # turn data into numpy ndarray
    min=data[:, 1:].min()
    max=data[:, 1:].max()
    print(f"the min and max value of data is {min} and {max} accordingly")
    np.random.shuffle(data)
    print("Pixel intensities are a number in: [1,255]; We will scale to [0,1]")
    X = data[:, 1:] / float(max) # data[:, 1:] / 255.0 normalize the data so it is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

class NaiveBayes:
    def fit (self, X,Y, smoothing=10e-3):
        self.gaussians = dict() #guassian parameters
        self.priors = dict() #guassian priors
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var' : current_x.var(axis=0) + smoothing
            }
            self.priors[c] = float(len(Y[Y==c]))/len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in iteritems(self.gaussians):
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)

if __name__ == '__main__':
    X, Y = get_data(10000)
    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = NaiveBayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))






