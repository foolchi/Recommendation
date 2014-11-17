import numpy as np
import matplotlib.pyplot as plt
from evaluation import evaluationRMSE

'''
Bias Recommendation
'''
class Bias:
    def __init__(self, data, test):
        self.data = data
        self.test = test
        maxs = self.data.max(0)
        self.nu = maxs[0] + 1
        self.ni = maxs[1] + 1

    def calculateBias(self):
        self.user_bias = np.zeros(self.nu)
        self.item_bias = np.zeros(self.ni)
        self.user_count = np.zeros(self.nu)
        self.item_count = np.zeros(self.ni)
        totalAverage = 0

        dataLength = len(self.data)
        for iter in range(dataLength):
            u = self.data[iter, 0]
            i = self.data[iter, 1]
            r = self.data[iter, 2]
            totalAverage += r
            self.user_bias[u] += r
            self.item_bias[i] += r
            self.user_count[u] += 1
            self.item_count[i] += 1
        self.user_bias /= np.where(self.user_count == 0, 1, self.user_count)
        self.item_bias /= np.where(self.item_count == 0, 1, self.item_count)
        # totalAverage /= dataLength
        # self.user_bias -= totalAverage
        # self.item_bias -= totalAverage
        # self.totalAverage = totalAverage

        self.user_bias_ratio = np.zeros(self.nu)
        for iter in range(dataLength):
            u = self.data[iter, 0]
            i = self.data[iter, 1]
            self.user_bias_ratio[u] += self.item_bias[i]
        self.user_bias_ratio /= np.where(self.user_count == 0, 1, self.user_count)
        self.user_bias_ratio = self.user_bias / np.where(self.user_bias_ratio == 0, 1, self.user_bias_ratio)
        self.defaultRate = np.average(self.item_bias)

    def predict(self, testData = None):
        if testData is not None:
            self.test = testData
        testSize = len(self.test)
        answer = self.test[:, 2]
        predicts = np.zeros(testSize)
        for iter in range(testSize):
            u = self.test[iter, 0]
            i = self.test[iter, 1]
            #predicts[iter] = self.user_bias[u]
            #predicts[iter] = self.item_bias[i]
            if u < self.nu and i < self.ni:
                predicts[iter] = self.user_bias_ratio[u] * self.item_bias[i]
                #predicts[iter] = self.user_bias[u]
                #predicts[iter] = self.totalAverage + self.user_bias[u] + self.item_bias[i]
            elif i < self.ni:
                predicts[iter] = self.item_bias[i]
            elif u < self.nu:
                predicts[iter] = self.user_bias_ratio[u]
            else:
                predicts[iter] = self.defaultRate
            if (predicts[iter] > 5):
                predicts[iter] = 5
            if (predicts[iter] < 1):
                predicts[iter] = 5
        return answer, predicts


def test_predict():
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    bias = Bias(np.loadtxt(fdata), np.loadtxt(ftest))
    bias.calculateBias()
    answer, predicts = bias.predict()
    err = evaluationRMSE(answer, predicts)
    print("Err: %f" % (err))

def test_bias():
    filen = "Data/100k/u.data"
    data = np.loadtxt(filen)

    maxs = data.max(0)
    print(maxs)
    nu = maxs[0]+1
    ni = maxs[1]+1

    user_bias = np.zeros(nu)
    item_bias = np.zeros(ni)
    user_count = np.zeros(nu)
    item_count = np.zeros(ni)

    for iteration in range(len(data)):
        u = data[iteration,0]
        i = data[iteration,1]
        r = data[iteration,2]
        user_bias[u] += r
        item_bias[i] += r
        user_count[u] += 1
        item_count[i] += 1

    user_bias /= np.where(user_count == 0, 1, user_count)
    item_bias /= np.where(item_count == 0, 1, item_count)
    print("user and item biases computed.")

    plt.figure()
    plt.subplot(211)
    plt.hist(user_bias, 100)
    plt.title('Utisateur')
    plt.subplot(212)
    plt.hist(item_bias, 100)
    plt.title('Item')
    plt.savefig('RecoBias.pdf')



if __name__ == '__main__':
    #test_biais()
    test_predict()