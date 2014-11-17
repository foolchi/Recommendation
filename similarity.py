import numpy as np
import pickle
import matplotlib.pyplot as plt
from evaluation import evaluationRMSE
import math
import matplotlib.pyplot as plt
from userBias import loadBias, loadItemInfo
'''
Collaborative Filter, Similarity
'''


folder = "Data/100k/"

class Similarity:
    def __init__(self, data = None, test = None):
        if data is not None and test is not None:
            self.data = data
            self.test = test
            maxs = self.data.max(0)
            self.nu = int(maxs[0] + 1)
            self.ni = int(maxs[1] + 1)
            self.sim = None
            self.den1 = None
            self.den2 = None
            self.user_bias = None
            self.calculateBias()

    def calculateBias(self):
        self.user_bias = np.zeros(self.nu)
        self.user_count = np.zeros(self.nu)
        dataLength = len(self.data)
        for iter in range(dataLength):
            u = self.data[iter, 0]
            r = self.data[iter, 2]
            self.user_bias[u] += r
            self.user_count[u] += 1
        self.user_bias /= np.where(self.user_count == 0, 1, self.user_count)

    def calcSimiMatrix(self):
        from svd import calcSim
        mat = np.zeros((self.nu, self.ni))
        for i in range(len(self.data)):
            currentData = self.data[i, :]
            user, item , rating = currentData[0], currentData[1], currentData[2]
            mat[user, item] = rating
        matSim = np.zeros((self.nu, self.nu))
        for u1 in range(self.nu):
            for u2 in range(u1, self.nu):
                #mU1 = mat[u1,]
                #currentSim = calcSim(mat[u1, :], mat[u2, :], self.user_bias[u1], self.user_bias[u2])
                currentSim = np.corrcoef(mat[u1, :], mat[u2, :])
                currentSim = currentSim[0,1]
                #currentSim = calcSim(mat[u1, :], mat[u2, :])
                matSim[u1, u2] = currentSim
                matSim[u2, u1] = currentSim
        self.sim = matSim


    def calcSimi(self):
        sim = np.zeros((self.nu, self.nu))
        den1 = np.zeros((self.nu, self.nu))
        den2 = np.zeros((self.nu, self.nu))
        for i in range(1, self.ni):
            if (i % 100 == 0):
                print(i)
            subdata = self.data[self.data[:, 1] == i, :]
            n = subdata.shape[0]
            for i1 in range(n):
                for i2 in range(i1, n):
                    u1 = subdata[i1, 0]
                    u2 = subdata[i2, 0]
                    r1 = subdata[i1, 2]
                    r2 = subdata[i2, 2]
                    v1 = r1 - self.user_bias[u1]
                    v1s = v1 ** 2
                    v2 = r2 - self.user_bias[u2]
                    v2s = v2 ** 2
                    simV = v1 * v2
                    sim[u1, u2] += simV
                    sim[u2, u1] += simV
                    den1[u1, u2] += v1s
                    den1[u2, u1] += v1s
                    den2[u1, u2] += v2s
                    den2[u2, u1] += v2s
        self.sim = sim / np.maximum(np.sqrt(den1) * np.sqrt(den2), 1)
        self.den1 = den1
        self.den2 = den2

    def saveSimi(self, n):
        sim_file = open(folder + "sim" + str(n) + ".pickle", 'wb')
        user_bias_file = open(folder + "user_bias" + str(n) + ".pickle", 'wb')
        den1_file = open(folder + "den1" + str(n) + ".pickle", 'wb')
        den2_file = open(folder + "den2" + str(n) + ".pickle", 'wb')
        if self.sim is not None:
            pickle.dump(self.sim, sim_file)
        if self.user_bias is not None:
            pickle.dump(self.user_bias, user_bias_file)
        if self.den1 is not None:
            pickle.dump(self.den1, den1_file)
        if self.den2 is not None:
            pickle.dump(self.den2, den2_file)
        sim_file.close()
        user_bias_file.close()
        den1_file.close()
        den2_file.close()

    def loadSimi(self, n):
        sim_file = open(folder + "sim" + str(n) + ".pickle", 'rb')
        user_bias_file = open(folder + "user_bias" + str(n) + ".pickle", 'rb')
        den1_file = open(folder + "den1" + str(n) + ".pickle", 'rb')
        den2_file = open(folder + "den2" + str(n) + ".pickle", 'rb')
        try:
            self.sim = pickle.load(sim_file)
            self.user_bias = pickle.load(user_bias_file)
            self.den1 = pickle.load(den1_file)
            self.den2 = pickle.load(den2_file)
        except:
            pass
        sim_file.close()
        user_bias_file.close()
        den1_file.close()
        den2_file.close()

    def predict(self, threshold = 0, testData = None):
        if testData is not None:
            self.test = testData
        answers = self.test[:, 2]
        testSize = len(answers)
        predicts = np.zeros(testSize)
        for i in range(testSize):
            # if (i % 1000 == 0):
            #     print(i)
            user = self.test[i, 0]
            item = self.test[i, 1]
            subdata = self.data[self.data[:, 1] == item, :]
            subdataSize = len(subdata)
            predicts[i] += self.user_bias[user]

            '''
            genre = itemInfo[item]
            for i in range(len(genre)):
                if (genre[i] != 0):
                    predicts[i] += 0.05 * userBias.getGenreBias(user,i)


            predicts[i] += 0.1 * userBias.getItemBias(item)
            '''

            sim_divisor = 0
            sim_dividen = 0
            for j in range(subdataSize):
                currentUser = subdata[j, 0]
                currentSim = self.sim[user, currentUser]
                if (math.fabs(currentSim) < threshold):
                    continue
                sim_divisor += math.fabs(currentSim)
                sim_dividen += currentSim * (subdata[j, 2] - self.user_bias[currentUser])
                #sim_dividen += currentSim * subdata[j, 2]
            if (sim_divisor != 0):
                predicts[i] += sim_dividen / sim_divisor
                if (predicts[i] < 1):
                    predicts[i] = 1
                elif (predicts[i] > 5):
                    predicts[i] = 5
        return answers, predicts

def test():
    from bias import Bias
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    simi = Similarity(np.loadtxt(fdata), np.loadtxt(ftest))
    #simi.calcSimi()
    simi.calcSimiMatrix()
    #simi.saveSimi(10)
    #simi.loadSimi(10)
    answer, predicts = simi.predict()
    err = evaluationRMSE(answer, predicts)
    print("Error: %f" % err)
    #simi.loadSimi(1)
    #print(simi.sim[:10, :10])

def simDistribution():
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    simi = Similarity(np.loadtxt(fdata), np.loadtxt(ftest))
    simi.loadSimi(10)
    simArray = []
    for u1 in range(simi.nu):
        for u2 in range(u1, simi.nu):
            simArray.append(simi.sim[u1, u2])
    plt.figure()
    plt.hist(simArray, 20)
    plt.show()


if __name__ == '__main__':
    test()