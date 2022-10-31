import unittest
import numpy as np

from Models import LogisticRegression
from sklearn.datasets import load_breast_cancer

class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        data = load_breast_cancer()
        self.x = data.data
        self.y = data.target

    def test_logistic_function(self):
        log_regr = LogisticRegression()
        x = np.random.rand(2000, 5)*100
        logistic_x = log_regr.logistic_function(x)
        assert x.shape == logistic_x.shape,"Shape should not change after logistic operation."
        assert ((logistic_x >= 0) | (logistic_x <= 1)).all(), "Logistic value should be between 0 and 1."

    def test_fit(self):
        log_regr = LogisticRegression(learning_rate=0.001,max_iter=1000)
        log_regr.fit(self.x,self.y)
        acc = log_regr.acc_score(self.x,self.y)
        assert acc > 0.5

    def test_weight_update(self):
        log_regr = LogisticRegression(learning_rate=0.001, max_iter=1)
        log_regr.fit(self.x, self.y)
        w_before = log_regr.weights
        log_regr.fit(self.x, self.y)
        w_after = log_regr.weights
        assert not np.array_equal(w_before,w_after)

if __name__ == '__main__':
    unittest.main()
