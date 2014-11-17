import numpy as np
import pickle
import matplotlib.pyplot as plt
from evaluation import evaluationRMSE
from evaluation import calcSim
import math

folder = "Data/100k/"

class Similarity:
    def __init__(self, fdata = None, ftest = None):
        if fdata is not None and ftest is not None:
            self.data = np.loadtxt(fdata)
            self.test = np.loadtxt(ftest)
            maxs = self.data.max(0)
            self.nu = int(maxs[0] + 1)
            self.ni = int(maxs[1] + 1)
            self.sim = None
            self.den1 = None
            self.den2 = None
            self.user_bias = None
            self.calculateBiais()

    def calculateBiais(self):
        self.item_bias = np.zeros(self.ni)
        self.item_count = np.zeros(self.ni)
        dataLength = len(self.data)
        for iter in range(dataLength):
            u = self.data[iter, 1]
            r = self.data[iter, 2]
            self.item_bias[u] += r
            self.item_count[u] += 1
        self.item_bias /= np.where(self.item_count == 0, 1, self.item_count)

    def calcSimiMatrix(self):
        mat = np.zeros((self.ni, self.nu))
        for i in range(len(self.data)):
            currentData = self.data[i, :]
            user, item , rating = currentData[0], currentData[1], currentData[2]
            mat[item, user] = rating
        matSim = np.zeros((self.ni, self.ni))
        for u1 in range(self.ni):
            for u2 in range(u1, self.ni):
                #mU1 = mat[u1,]
                currentSim = calcSim(mat[u1, :], mat[u2, :], self.item_bias[u1], self.item_bias[u2])
                #currentSim = calcSim(mat[u1, :], mat[u2, :])
                matSim[u1, u2] = currentSim
                matSim[u2, u1] = currentSim
        self.sim = matSim

    def predict(self, threshold = 0.5, testData = None):
        if testData is not None:
            self.test = testData
        answer = self.test[:, 2]
        testSize = len(answer)
        predicts = np.zeros(testSize)
        for i in range(testSize):
            # if (i % 1000 == 0):
            #     print(i)
            user = self.test[i, 0]
            item = self.test[i, 1]
            subdata = self.data[self.data[:, 0] == user, :]
            subdataSize = len(subdata)
            predicts[i] += self.item_bias[item]
            sim_divisor = 0
            sim_dividen = 0
            for j in range(subdataSize):
                currentItem = subdata[j, 1]
                currentSim = self.sim[item, currentItem]
                if (math.fabs(currentSim) < threshold):
                    continue
                sim_divisor += math.fabs(currentSim)
                sim_dividen += currentSim * (subdata[j, 2] - self.item_bias[currentItem])
            if (sim_divisor != 0):
                predicts[i] += sim_dividen / sim_divisor
                if (predicts[i] <= 0):
                    predicts[i] = 0
                elif (predicts[i] > 5):
                    predicts[i] = 5
        return answer, predicts


def test():
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    simi = Similarity(fdata, ftest)
    #simi.calcSimi()
    #simi.calcSimiLocal()
    #simi.saveSimi(1)
    simi.calcSimiMatrix()
    answer, predicts = simi.predict()
    err = evaluationRMSE(answer, predicts)
    print("Error: %f" % err)
    #simi.loadSimi(1)
    #print(simi.sim[:10, :10])


if __name__ == '__main__':
    #test_simi()
    test()