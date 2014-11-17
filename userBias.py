import numpy as np
import pickle

class UserBias:
    def __init__(self, udata):
        self.udata = udata
        maxs = self.udata.max(0)
        self.nu = int(maxs[0] + 1)
        self.ni = int(maxs[1] + 1)

    def generateBias(self):
        self.itemInfo = loadItemInfo()
        self.userBias = np.zeros(self.nu)
        self.userCount = np.zeros(self.nu)
        self.itemBias = np.zeros(self.ni)
        self.itemCount = np.zeros(self.ni)
        self.genreBias = np.zeros((self.nu, 19))
        self.genreBiasCount = np.zeros((self.nu, 19))
        self.averageRating = 0

        dataSize = len(self.udata)
        for i in range(dataSize):
            currentData = self.udata[i, :]
            user = currentData[0]
            item = currentData[1]
            rating = currentData[2]
            self.averageRating += rating
            self.userBias[user] += rating
            self.userCount[user] += 1
            self.itemBias[item] += rating
            self.itemCount[item] += 1
            itemGenre = self.itemInfo[item, :]
            for index, genre in enumerate(itemGenre):
                if genre == 0:
                    continue
                self.genreBias[user, index] += rating
                self.genreBiasCount[user, index] += 1
        self.averageRating /= dataSize
        self.userBias /= np.where(self.userCount == 0, 1, self.userCount)
        self.genreBias /= np.where(self.genreBiasCount == 0, 1, self.genreBiasCount)
        self.itemBias /= np.where(self.itemCount == 0, 1, self.itemCount)

    def getGenreBias(self, user, genre):
        if (self.genreBias[user, genre] == 0):
            return 0
        return self.genreBias[user, genre] - self.userBias[user]

    def getItemBias(self, item):
        if (self.itemBias[item] == 0):
            return self.itemBias[item]
        return self.itemBias[item] - self.averageRating

    def getUserBias(self, user):
        return self.userBias[user]


def generateItemInfo(itemFile = None):
    if (itemFile is None):
        itemFile = "Data/100k/u.item"
    nGenre = 19
    nItem = 1683
    itemInfo = np.zeros((nItem, nGenre))
    with open(itemFile) as f:
        line = f.readline()
        while(line is not None):
            splits = line.split('|')
            if (len(splits) != 24):
                break
            item = int(splits[0])
            print(item)
            for i in range(5, 24):
                #print(int(splits[i]))
                itemInfo[item, i-5] = int(splits[i])
            line = f.readline()
            print(line)
    itemInfoFile = open("Data/100k/itemInfo.pickle", 'wb')
    pickle.dump(itemInfo, itemInfoFile)
    return itemInfo

def loadItemInfo(pickleFile = None):
    if (pickleFile is None):
        pickleFile = "Data/100k/itemInfo.pickle"
    return pickle.load(open(pickleFile, 'rb'))

def testItemInfo():
    itemInfo = loadItemInfo()
    print(itemInfo[2,:])


def loadBias():
    fdata = "Data/100k/u1.base"
    data = np.loadtxt(fdata)
    userBias = UserBias(data)
    userBias.generateBias()
    return userBias

def testBias():
    userBias = loadBias()
    for i in range(19):
        print(userBias.getGenreBias(1, i))



if __name__ == '__main__':
    #generateItemInfo("Data/100k/u.item")
    #loadItemInfo()
    #loadBias()
    #testItemInfo()
    testBias()