from bias import Bias
from evaluation import evaluationRMSE
from similarity import Similarity
from svd import SVD
from matFactory import MatFactory
import numpy as np
from combination import Combination

'''
Test for all database
'''

folder = "Data/100k/"

class CalAll:
    def __init__(self, data, test):
        self.data = data
        self.test = test

    def calAll(self):
        self.errs = [0] * 5
        bias = Bias(self.data, self.test)
        bias.calculateBias()
        answers, predicts = bias.predict()
        err = evaluationRMSE(answers, predicts)
        self.errs[0] = err
        print("Bias: %f" % err)

        similarity = Similarity(self.data, self.test)
        similarity.calculateBias()
        similarity.calcSimiMatrix()
        answers, predicts = similarity.predict()
        err = evaluationRMSE(answers, predicts)
        self.errs[1] = err
        print("Similarity: %f" % err)

        svd = SVD(self.data, self.test)
        svd.generaterMat()
        svd.calcSVD()
        answers, predicts = svd.predict()
        err = evaluationRMSE(answers, predicts)
        self.errs[2] = err
        print("SVD: %f" % err)

        matFactory = MatFactory(self.data, self.test)
        matFactory.train(20, 35)
        answers, predicts = matFactory.predict()
        err = evaluationRMSE(answers, predicts)
        self.errs[3] = err
        print("MatFactory: %f" % evaluationRMSE(answers, predicts))

        combination = Combination(self.data)
        combination.separateData()
        combination.calculate()
        combination.train(alpha = 0.01, iter = 10000)
        answers, predicts = combination.predict(self.test)
        err = evaluationRMSE(answers, predicts)
        self.errs[4] = err
        print("Combination: %f" % err)
        return self.errs


def test():
    errs = np.zeros((5,5))
    for i in range(1, 6):
        print("================DataBase: ", i, "================")
        dataFile = folder + "u" + str(i) + ".base"
        testFile = folder + "u" + str(i) + ".test"
        calAll = CalAll(np.loadtxt(dataFile), np.loadtxt(testFile))
        errs[i-1, :] = calAll.calAll()
    errs = errs.T
    print("================DONE================")
    for i in range(5):
        currentErr = errs[i, :]
        print(currentErr, currentErr.mean())


if __name__ == '__main__':
    test()
