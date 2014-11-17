import numpy as np
import pickle
import matplotlib.pyplot as plt
from evaluation import evaluationRMSE
import math
import matplotlib.pyplot as plt
from userBias import loadBias, loadItemInfo

'''
Collaborative Filter, KNN
'''

folder = "Data/100k/"

class KNN:
    def __init__(self, data = None, test = None):
        if data is not None and test is not None:
            self.data = data
            self.test = test
            maxs = self.data.max(0)
            self.nu = int(maxs[0] + 1)
            self.ni = int(maxs[1] + 1)
            self.sim = None
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

    def calcSimMatrix(self):
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

    def saveSim(self, n):
        sim_file = open(folder + "sim" + str(n) + ".pickle", 'wb')
        user_bias_file = open(folder + "user_bias" + str(n) + ".pickle", 'wb')
        if self.sim is not None:
            pickle.dump(self.sim, sim_file)
        if self.user_bias is not None:
            pickle.dump(self.user_bias, user_bias_file)
        sim_file.close()
        user_bias_file.close()

    def loadSim(self, n):
        sim_file = open(folder + "sim" + str(n) + ".pickle", 'rb')
        user_bias_file = open(folder + "user_bias" + str(n) + ".pickle", 'rb')
        try:
            self.sim = pickle.load(sim_file)
            self.user_bias = pickle.load(user_bias_file)
        except:
            pass
        sim_file.close()
        user_bias_file.close()

    def predict(self, testData = None, n = None):
        if testData is not None:
            self.test = testData
        if n is None:
            n = 10
        answers = self.test[:, 2]
        testSize = len(answers)
        predicts = np.zeros(testSize)
        for i in range(testSize):
            # if (i % 1000 == 0):
            #     print(i)
            user = self.test[i, 0]
            item = self.test[i, 1]
            subdata = self.data[self.data[:, 1] == item, :]
            #print(subdata)
            subdataSize = len(subdata)
            if (subdataSize > n):
                subUsers = subdata[:, 0]
                subSim = [self.sim[user, subUser] for subUser in subUsers]
                #print(subSim)
                subSim.sort(reverse = True)
                simThre = subSim[n - 1]
                validUser = [subUser for subUser in subUsers if self.sim[user, subUser] >= simThre]
                #print(validUser)
                knnData = []
                for d in subdata:
                    if d[0] in validUser:
                        knnData.append(d)
                subdata = np.array(knnData)
                #print(subdata)
                subdataSize = len(subdata)
                #print(subdataSize)
            predicts[i] += self.user_bias[user]

            sim_divisor = 0
            sim_dividen = 0
            for j in range(subdataSize):
                currentUser = subdata[j, 0]
                currentSim = self.sim[user, currentUser]
                sim_divisor += math.fabs(currentSim)
                sim_dividen += currentSim * (subdata[j, 2] - self.user_bias[currentUser])
            if (sim_divisor != 0):
                predicts[i] += sim_dividen / sim_divisor
                if (predicts[i] < 1):
                    predicts[i] = 1
                elif (predicts[i] > 5):
                    predicts[i] = 5
        return answers, predicts

def test_knn():
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    knn = KNN(np.loadtxt(fdata), np.loadtxt(ftest))
    #knn.calcSimMatrix()
    #knn.saveSim(100)
    knn.loadSim(100)
    answer, predicts = knn.predict(n = 15)
    err = evaluationRMSE(answer, predicts)
    print("Error: %f" % err)


if __name__ == '__main__':
    test_knn()