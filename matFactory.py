import numpy as np
import matplotlib.pyplot as plt
from evaluation import evaluationRMSE

class MatFactory:
    def __init__(self, data = None, test = None):
        if data is not None and test is not None:
            self.data = data
            self.test = test
            maxs = self.data.max(0)
            self.nu = int(maxs[0] + 1)
            self.ni = int(maxs[1] + 1)

    def train(self, epochs = None, nZ = None):
        random = np.random.RandomState(0)
        if (epochs is None):
            epochs = 5
        if (nZ is None):
            nZ = 10
        l1_weight = 0.00
        l2_weight = 0.0001
        learning_rate = 0.01

        train_indexes = np.arange(len(self.data))

        user_latent = np.random.randn(self.nu, nZ)
        item_latent = np.random.randn(self.ni, nZ)
        user_latent = np.where(user_latent > 0, user_latent, 0)
        item_latent = np.where(item_latent > 0, item_latent, 0)

        for epoch in range(epochs):
            learning_rate *= 0.9
            # Update
            random.shuffle(train_indexes)
            for index in train_indexes:
                label, user, item = self.data[index,2], self.data[index,0], self.data[index,1]
                gamma_u, gamma_i = user_latent[user, :], item_latent[item, :]
                #print(gamma_u.shape, gamma_i.shape)
                delta_label = 2 * (label - np.dot(gamma_u, gamma_i))
                gradient_u = l2_weight * gamma_u + l1_weight - delta_label * gamma_i
                gamma_u_prime = gamma_u - learning_rate * gradient_u
                user_latent[user, :] = np.where(gamma_u_prime * gamma_u > 0, gamma_u_prime, 0)
                gradient_i = l2_weight * gamma_i + l1_weight - delta_label * gamma_u
                gamma_i_prime = gamma_i - learning_rate * gradient_i
                item_latent[item, :] = np.where(gamma_i_prime * gamma_i > 0, gamma_i_prime, 0)
        self.user_latent = user_latent
        self.item_latent = item_latent

    def predict(self, testData = None):
        if testData is not None:
            self.test = testData
        answer = self.test[:, 2]
        testSize = len(answer)
        predicts = np.zeros(testSize)

        for i in range(testSize):
            user, item = self.test[i, 0], self.test[i, 1]
            if user < self.nu and item < self.ni:
                predicts[i] = np.dot(self.user_latent[user, :], self.item_latent[item, :])
            if (predicts[i] > 5):
                predicts[i] = 5
            if (predicts[i] < 1):
                predicts[i] = 1
        return answer, predicts


    def plot(self):
        plt.figure()
        plt.imshow(self.user_latent[:100,:], interpolation="nearest")
        plt.colorbar()
        plt.savefig("userLatent.pdf")


def test_matFactory():
    fdata = "Data/100k/u1.base"
    ftest = "Data/100k/u1.test"
    matFactory = MatFactory(np.loadtxt(fdata), np.loadtxt(ftest))
    factors = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200]
    for i in factors:
        matFactory.train(20, i)
        answer, predicts = matFactory.predict()
        err = evaluationRMSE(answer, predicts)
        print("Factors: %d, Error: %f" % (i, err))
    #matFactory.plot()

if __name__ == '__main__':
    test_matFactory()