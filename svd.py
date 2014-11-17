import numpy as np
import math
from evaluation import evaluationRMSE
from evaluation import calcSim

class SVD:
    def __init__(self, data = None, test = None):
        if data is not None and test is not None:
            self.data = data
            self.test = test
            maxs = self.data.max(0)
            self.nu = int(maxs[0] + 1)
            self.ni = int(maxs[1] + 1)
            self.calculateBiais()

    def calculateBiais(self):
        self.user_bias = np.zeros(self.nu)
        self.user_count = np.zeros(self.nu)
        dataLength = len(self.data)
        for iter in range(dataLength):
            u = self.data[iter, 0]
            r = self.data[iter, 2]
            self.user_bias[u] += r
            self.user_count[u] += 1
        self.user_bias /= np.where(self.user_count == 0, 1, self.user_count)

    def generaterMat(self):
        self.mat = np.zeros((self.nu, self.ni))
        dataSize = len(self.data)
        for i in range(dataSize):
            currentLine = self.data[i, :]
            user = currentLine[0]
            item = currentLine[1]
            rating = currentLine[2]
            self.mat[user, item] = rating

    def calcSVD(self, threshold = None):
        if (threshold is None):
            threshold = 0.95
        U, Sigma, VT = np.linalg.svd(self.mat)
        self.U = U
        self.Sigma = np.zeros(self.mat.shape)
        for i in range(Sigma.shape[0]):
            self.Sigma[i, i] = Sigma[i]
        self.VT = VT
        #print(Sigma.shape, self.U.shape, self.Sigma.shape, self.VT.shape)

        sigma2 = Sigma ** 2
        threshold *= sum(sigma2)
        currentSum = 0
        adjustSize = Sigma.shape[0]
        for i in range(Sigma.shape[0]):
            currentSum += sigma2[i]
            if (currentSum > threshold):
                adjustSize = i
                break
        #print(adjustSize)
        adjustSigma = np.mat(np.eye(adjustSize) * Sigma[:adjustSize])
        #xFormedUsers = (U[:, :adjustSize].dot(adjustSigma)).dot(VT[:adjustSize,:])
        xFormedUsers = (self.mat.dot(VT[:, :adjustSize])).dot(adjustSigma.I)
        self.matSVD = xFormedUsers

        #xFormedUserBias = np.average(xFormedUsers, 1)
        #for i in range(self.nu):
        #    xFormedUsers[i, :] -= xFormedUserBias[i]
        sim = np.zeros((self.nu, self.nu))
        for u1 in range(1, self.nu):
            for u2 in range(u1, self.nu):
                #currentSim = calcSim(xFormedUsers[u1, :], xFormedUsers[u2, :], xFormedUserBias[u1], xFormedUserBias[u2])
                currentSim = np.corrcoef(xFormedUsers[u1, :], xFormedUsers[u2, :])
                sim[u1, u2] = currentSim[0,1]
                sim[u2, u1] = currentSim[0,1]
        self.sim = sim


    def predict(self, threshold = 0, testData = None):
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

            '''
            predicts[i] = self.matSVD[user, item]
            if (predicts[i] <= 0):
                predicts[i] = 0
            elif (predicts[i] > 5):
                predicts[i] = 5
                '''
            predicts[i] = self.user_bias[user]
            #predicts[i] += self.U[user, :].dot(self.Sigma).dot(self.VT[:, item])
            #predicts[i] += self.U[user, :].dot(self.VT[:, item])
            subdata = self.data[self.data[:, 1] == item, :]
            subdataSize = len(subdata)
            sim_divisor = 0
            sim_dividen = 0
            for j in range(subdataSize):
                currentUser = subdata[j, 0]
                currentSim = self.sim[user, currentUser]
                if (math.fabs(currentSim) < threshold):
                    continue
                sim_divisor += math.fabs(currentSim)
                sim_dividen += currentSim * (subdata[j, 2] - self.user_bias[currentUser])
            if (sim_divisor != 0):
                predicts[i] += sim_dividen / sim_divisor
            if (predicts[i] < 1):
                predicts[i] = 1
            elif (predicts[i] > 5):
                predicts[i] = 5

        return answer, predicts


def test_SVD():
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    svd = SVD(np.loadtxt(fdata), np.loadtxt(ftest))
    svd.generaterMat()
    svd.calcSVD()
    answer, predicts = svd.predict()
    err = evaluationRMSE(answer, predicts)
    print("Error: %f" % err)


if __name__ == '__main__':
    test_SVD()