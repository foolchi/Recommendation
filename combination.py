from bias import Bias
from similarity import Similarity
from svd import SVD
from matFactory import MatFactory
import numpy as np
from evaluation import evaluationRMSE
import pickle
from math import sqrt

'''
Combination Of Algorithms
'''

predictsFile = "Data/100k/combinationPredicts.pickle"
classFile = "Data/100k/combinationClass.pickle"

class Combination:
    def __init__(self, data):
        self.data = data

    def separateData(self, mod = None):
        if mod is None:
            mod = 2
        self.trainData = []
        self.testData = []
        for i, d in enumerate(self.data):
            if i % 5 == mod:
                self.testData.append(d)
            else:
                self.trainData.append(d)
        self.trainData = np.array(self.trainData)
        self.testData = np.array(self.testData)
        self.testSize = len(self.testData)
        self.answers = self.testData[:, 2]


    def calculate(self):
        self.allPredicts = np.zeros((4, self.testSize))

        bias = Bias(self.trainData, self.testData)
        bias.calculateBias()
        answers, predicts = bias.predict()
        self.biasClass = bias
        self.allPredicts[0, :] = predicts
        #print("Bias: %f" % evaluationRMSE(answers, predicts))

        similarity = Similarity(self.trainData, self.testData)
        similarity.calculateBias()
        similarity.calcSimiMatrix()
        answers, predicts = similarity.predict()
        self.similarityClass = similarity
        self.allPredicts[1, :] = predicts
        #print("Similarity: %f" % evaluationRMSE(answers, predicts))

        svd = SVD(self.trainData, self.testData)
        svd.generaterMat()
        svd.calcSVD()
        answers, predicts = svd.predict()
        self.svdClass = svd
        self.allPredicts[2, :] = predicts
        #print("SVD: %f" % evaluationRMSE(answers, predicts))

        matFactory = MatFactory(self.trainData, self.testData)
        matFactory.train(10, 11)
        answers, predicts = matFactory.predict()
        self.matFactoryClass = matFactory
        self.allPredicts[3, :] = predicts
        #print("MatFactory: %f" % evaluationRMSE(answers, predicts))

        pickleFile = open(predictsFile, 'wb')
        pickle.dump(self.allPredicts, pickleFile)

    def loadPredicts(self, file = None):
        if file is None:
            file = predictsFile
        pickleFile = open(predictsFile, 'rb')
        self.allPredicts = pickle.load(pickleFile)

    def train(self, alpha = None, iter = None):
        if alpha is None:
            alpha = 0.01
        if iter is None:
            iter = 20
        self.theta = np.zeros((1, 5))
        self.theta[0, 1:] = 0.25

        self.predicts = np.zeros((5, self.testSize))
        self.predicts[1: , :] = self.allPredicts
        self.predicts[0, :] = 1

        for i in range(iter):

            diffMat = self.theta.dot(self.predicts) - self.answers
            #diffMat *= diffMat
            #diffMat = np.sqrt(diffMat)
            #print("Iteration: ", i, ", error: ", diffMat.sum(), ", theta: ", self.theta)
            self.theta[0, 0] -= alpha * diffMat.mean()
            for k in range(1, 5):
                self.theta[0, k] -= alpha * (diffMat * self.predicts[k, :]).mean()
        print("Theta: ", self.theta)
    def predict(self, test):
        predictsMat = np.zeros((5, len(test)))
        predictsMat[0, :] = 1
        _, predictsMat[1, :] = self.biasClass.predict(testData = test)
        _, predictsMat[2, :] = self.similarityClass.predict(testData = test)
        _, predictsMat[3, :] = self.svdClass.predict(testData = test)
        _, predictsMat[4, :] = self.matFactoryClass.predict(testData = test)
        predicts = self.theta.dot(predictsMat)
        predicts = np.where(predicts > 5, 5, predicts)
        predicts = np.where(predicts < 1, 1, predicts)
        answers = test[:, 2]
        return answers, predicts.T

def test():
    fdata = "Data/100k/u1.base"
    data = np.loadtxt(fdata)
    combination = Combination(data)
    combination.separateData()
    combination.calculate()

def saveCombinationClass():
    fdata = "Data/100k/u1.base"
    data = np.loadtxt(fdata)
    combination = Combination(data)
    combination.separateData()
    combination.calculate()
    pickleFile = open(classFile, 'wb')
    pickle.dump(combination, pickleFile)

def loadCombinationClass():
    pickleFile = open(classFile, 'rb')
    return pickle.load(pickleFile)

def testTrain():
    #saveCombinationClass()
    combination = loadCombinationClass()
    combination.train(alpha = 0.01, iter = 10000)
    tdata = "Data/100k/u1.test"
    answers, predicts = combination.predict(np.loadtxt(tdata))
    err = evaluationRMSE(answers, predicts)
    print("Error : %f" % err)


if __name__ == '__main__':
    #test()
    testTrain()
