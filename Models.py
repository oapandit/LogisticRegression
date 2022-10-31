import random

import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    '''
    Implementation of logistic regression for binary classes. It models input feature vector x and output class y with
    p(y;w,b) = logistic_function( w.T *x + b) where w is weight vector and b is bias term and logistic_function is given as
    logistic_function(x) = 1/(1+ e^(-x)).

    While training the model following binary cross entropy loss is considered
    loss = - (y.log(p(y;w,b)) + (1-y)y.log(1-p(y;w,b)))

    :param reg: It defines the regularization used while training the model. Valid regularizers are None, l1, or l2.
    :param max_iter: Number of iteration to update weights.
    :param learning_rate: Learning rate is a positive float which decides the rate of update of weights at each
    iteration.
    :param regu_penalty: If regularization is used this parameter decides the rate of update of weights
    :param batch_size: This decides the number of input samples used for updating weights, if it is set to 1 the
    algorithm uses stochastic gradient descent, if batch size is less than total samples it will be trained in mini
    batch gradient descent otherwise with general gradient descent approach.
    '''

    def __init__(self, learning_rate=0.01, reg='l2', regu_penalty=0.1, max_iter=100, batch_size=1, is_verbose=False):
        self.learning_rate = learning_rate
        self.regu_penalty = regu_penalty
        self.reg = reg
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.is_verbose = is_verbose
        self.weights = None
        self.check_params_format()

    def check_params_format(self):
        if (not isinstance(self.learning_rate, float)) or self.learning_rate < 0:
            raise ValueError(f'Learning rate {self.learning_rate} should be positive float.')
        if (not isinstance(self.regu_penalty, float)) or self.regu_penalty < 0:
            raise ValueError(f'Regulizer penalty {self.regu_penalty} should be positive float.')
        if self.reg is not None and self.reg not in ['l1', 'l2']:
            raise ValueError(f'Regulizer {self.reg} is not valid; valid regulizers: None, l1, or l2.')
        if (not isinstance(self.max_iter, int)) or self.max_iter < 0:
            raise ValueError(f'Max iteration {self.max_iter} should be positive integer.')
        if self.batch_size is not None and ((not isinstance(self.batch_size, int)) or self.batch_size <= 0):
            raise ValueError(f'Batch size {self.batch_size} should be strictly positive integer.')

    def convert_to_numpy(self, data):
        try:
            data = np.array(data, dtype=np.float32)
        except ValueError:
            raise ValueError("Unable to convert data to numpy dtype.")
        return data

    def add_interecept(self, x):
        num_samples, features = x.shape
        intercept = np.ones((num_samples, 1))
        x = np.concatenate((intercept, x), axis=1)
        assert num_samples == x.shape[0] and features + 1 == x.shape[1]
        return x

    def logistic_function(self, data):
        return 1 / (1 + np.exp(-data))

    def loss(self, y_true, logistic_h):
        loss = -np.average((y_true * np.log(logistic_h) + (1 - y_true) * np.log((1 - logistic_h))))
        return loss

    def data_conversion(self, x, y):
        x, y = self.convert_to_numpy(x), self.convert_to_numpy(y)
        assert len(x.shape) == 2, "Given feature vector is not 2-dimensional."
        assert x.shape[0] == y.shape[0], "Features and label dimensions do not match."
        assert ((y==0) | (y==1)).all(),"Output contains labels other than 0 or 1."
        y = y.reshape(-1, 1)
        return x, y

    def fit(self, x, y):
        x, y = self.data_conversion(x, y)
        x = self.add_interecept(x)
        num_features = x.shape[1]
        self.weights = np.random.normal(size=(num_features, 1))
        for i in range(self.max_iter):
            h = np.dot(x, self.weights)
            logistic_h = self.logistic_function(h)
            loss = self.loss(y, logistic_h)
            if self.is_verbose and i%5==0:
                print(f'Iteration: {i} --- loss: {round(loss,4)}, accuracy: {self.acc_score(x,y)}')
            self.update_weights(x, y, logistic_h)

    def loss_gradient_wrt_w(self, x, y_true, logistic_h):
        n_samples = x.shape[0]
        if self.batch_size is None or self.batch_size >= n_samples: # gradient descent
            loss_gradient_wrt_w = np.average(x * (logistic_h - y_true), axis=0).reshape(-1, 1)
        elif self.batch_size == 1: # stochastic gradient descent
            sample_ind = random.randint(0, n_samples - 1)
            loss_gradient_wrt_w = (x[sample_ind] * (logistic_h[sample_ind] - y_true[sample_ind])).reshape(-1, 1)
        elif self.batch_size > 1: # minibatch gradient descent
            sample_inds = random.sample(list(range(n_samples)), self.batch_size)
            x_trunc = np.take(x, sample_inds, axis=0)
            logistic_h_trunc = np.take(logistic_h, sample_inds, axis=0)
            y_true_trunc = np.take(y_true, sample_inds, axis=0)
            loss_gradient_wrt_w = np.average(x_trunc * (logistic_h_trunc - y_true_trunc), axis=0).reshape(-1, 1)
        else:
            raise ValueError(f'Batch size {self.batch_size} is not implemented.')
        return loss_gradient_wrt_w

    def update_weights(self, x, y_true, logistic_h):
        total_gradient = self.loss_gradient_wrt_w(x, y_true, logistic_h) + self.regulizer_grad(self.weights)
        self.weights -= self.learning_rate * total_gradient

    def regulizer_grad(self, vec):
        if self.reg is None:
            reg_grad = 0
        elif self.reg == 'l1':
            reg_grad = np.sign(vec)
        elif self.reg == 'l2':
            reg_grad = 2 * vec
        else:
            raise NotImplementedError(f'Regularizer {self.reg} is not implemented.')
        return self.regu_penalty * reg_grad

    def calculate_h(self, x):
        '''
        h = wT.x + b
        '''
        assert self.weights is not None, "model is not trained."
        x = self.convert_to_numpy(x)
        if x.shape[1] + 1 == self.weights.shape[0]:
            x = self.add_interecept(x)
        h = np.dot(x, self.weights)
        return h

    def predict_proba(self, x):
        '''
        h = wT.x + b
        This score is passed through logistic function to get probability of class 1.
        '''
        h = self.calculate_h(x)
        pred_prob = self.logistic_function(h)
        return pred_prob

    def predict(self, x):
        pred_scores = self.calculate_h(x)
        preds = np.where(pred_scores >= 0, 1, 0)
        return preds

    def acc_score(self, x, y):
        x, y = self.data_conversion(x, y)
        y_pred = self.predict(x)
        return accuracy_score(y, y_pred)
