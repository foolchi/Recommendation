from math import sqrt
import numpy as np


def evaluationRMSE(data1, data2):
    #print("Data1: ", data1.shape, ", Data2: ", data2.shape)
    if (len(data1) != len(data2)):
        raise("Length not equal")
    dataSize = len(data1)
    if (dataSize == 0):
         raise("Data is empty")
    # with open('diffResult.txt', 'w') as f:
    #     for i in range(len(data1)):
    #         f.write(str(data1[i]))
    #         f.write('\t')
    #         f.write(str(data2[i]))
    #         f.write('\n')
    data1 = np.mat(data1)
    if (data1.shape[1] != 1):
        data1 = data1.T
    data2 = np.mat(data2)
    if (data2.shape[1] != 1):
        data2 = data2.T

    diff = data1 - data2
    diff = diff.T * diff
    #print(diff.shape)

    return sqrt(diff.sum() / dataSize)

def calcSim(p1, p2, p1av = None, p2av = None):
    p1 = np.where(p2 == 0, 0, p1)
    p2 = np.where(p1 == 0, 0, p2)
    numCommon = np.where(p1 == 0, 0, 1)
    if (numCommon.sum() < 2):
        return 0

    if (len(p1) < 2):
        return 0
    if (p1av is not None):
        p1 = np.where(p1 == 0, 0, p1 - p1av)
    if (p2av is not None):
        p2 = np.where(p2 == 0, 0, p2 - p2av)
    num = p1.dot(p2.T)
    denom = np.linalg.norm(p1) * np.linalg.norm(p2)
    if (denom == 0):
        return 0
    return num / denom

def test_sim():
    p1 = [1,2,3,0]
    p2 = [4,5,0,6]
    p1av = sum(p1)/3
    p2av = sum(p2)/3
    print(calcSim(p1, p2, p1av, p2av))

if __name__ == '__main__':
    test_sim()