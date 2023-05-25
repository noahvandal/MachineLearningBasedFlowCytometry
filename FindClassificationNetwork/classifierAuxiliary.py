import os 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


torch.manual_seed(43)
np.random.seed(43)
random.seed(43)


def createDataset(path):  #input path to folder containing train; ensure all samples have name corresponding to class in them. 
    print('path',path)
    datalist = os.listdir(path)
    allpaths = []
    for data in datalist:
        classtype = isClass(data)
        allpaths.append([path + data, classtype])
    allpaths = pd.DataFrame(allpaths, columns=["Image", "Class"])
    return allpaths

def getDataset(path, batchsize,numsteps,isVal, transformList=None):
    # imgdata = createDataset(path)
    imgdata = dataGenerator(path, batchsize,numsteps,isVal, transformList)
    loadedData = DataLoader(imgdata,batchsize,shuffle=True)
    return loadedData

def isClass(str): ## given string, is the class contained in the name? 
    stringClass = ""
    if "HPNE"  in str:
        # stringClass = 'HPNE'
        stringClass = [1,0]
    if 'MIA' in str:
        # stringClass = 'MIA'
        stringClass = [0,1]
    return stringClass


class dataGenerator(Dataset):
    def __init__(self, dataframe, batchsize, numsteps, isVal = False, transformList=None):
        self.dataframe = dataframe
        self.batchsize = batchsize
        self.numsteps = numsteps
        self.Validation = isVal
        self.resize = (64,64)
        self.transformList = transformList

    
    def __len__(self):
        length = len(self.dataframe) 
        # print(length)
        return length

    def on_epoch_end(self):
        self.dataframe = self.dataframe.reset_index(drop = True)

    def __getitem__(self, index):
        images = []
        labels = []

        img = cv2.imread(self.dataframe["Image"][index])
        # print(self.dataframe["Image"][index])
        img = cv2.resize(img,self.resize)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = np.expand_dims(img, axis=2)
        label = np.array(self.dataframe["Class"][index])

        images.append(img)
        labels.append(label)

        if self.transformList:
            # plt.imshow(img)
            # plt.show()
            img = np.transpose(img, [2, 0, 1]) ## torch expects [:.., W, H]
            img = torch.from_numpy(img)
            img = self.transformList(img)
            img = img.float()

            # testshow = np.asarray(img)
            # testshow = np.transpose(testshow, [1, 2, 0])
            # plt.imshow((testshow).astype(np.uint8))
            # plt.show()   


        else:
            img = np.transpose(img, [2, 0, 1]) ## torch expects [:.., W, H]
            img = torch.from_numpy(img)
            img = img.float()

        label = torch.from_numpy(label)
        label = label.float()     


        if self.Validation:
            return (img, label, self.dataframe['Image'][index])

        else:
            return (img, label, self.dataframe['Image'][index])
        

def checkAccuracy(tsr1, tsr2, classesPresent, printOutput):
    areEqual = False
    accCount = 0
    accuracy = 0
    compareList = []

    if len(tsr1) == len(tsr2):
        if printOutput:
            print('equal!!')
        areEqual = True

    if areEqual:
        for i in range(0,len(tsr1)):
            if tsr1[i] == tsr2[i]:
                if printOutput:
                    print('Equal!', classesPresent[tsr1[1]])
                accCount += 1
            if tsr1[i] != tsr2[i]:
                if printOutput:
                    print('Unequal: Predict:', classesPresent[tsr1[i]], 'Actual:', classesPresent[tsr2[i]])
            
            compareList.append([classesPresent[tsr1[i]],classesPresent[tsr2[i]]])
        
        accuracy = accCount / len(tsr1)

    
    return accuracy, compareList


def datasetAcquirerShuffler(srcPath, numTrain, numVal):  ## input path to dataset, and will split into test and train by itself, using random shuffle.
    allFiles = os.listdir(srcPath)

    for i in range(7):  ## shuffle 3 times
        random.shuffle(allFiles) ## where the shuffle takes place!!

    trainPaths = []
    valPaths = []
    for i, data in enumerate(allFiles):
        if i < numTrain:
            classtype = isClass(data)
            trainPaths.append([srcPath + data, classtype])
        if numTrain <= i < (numTrain + numVal):
            classtype = isClass(data)
            valPaths.append([srcPath + data, classtype])
    trainPaths = pd.DataFrame(trainPaths, columns=["Image", "Class"])
    valPaths = pd.DataFrame(valPaths, columns = ['Image','Class'])
    return trainPaths, valPaths

def createDataFrameFolder(srcPath):
    allFiles = os.listdir(srcPath)
    allPaths = []
    for i, data in enumerate(allFiles):
        classtype = isClass(data)
        allPaths.append([srcPath + data, classtype])
    allPaths = pd.DataFrame(allPaths, columns=["Image", "Class"])
    return allPaths


def datasetAcquirerShufflerWithTest(srcPath, numTrain, numVal, numTest):  ## input path to dataset, and will split into test and train by itself, using random shuffle.
    allFiles = os.listdir(srcPath)

    for i in range(43 + 6):  ## shuffle 3 times
        random.shuffle(allFiles) ## where the shuffle takes place!!

    trainPaths = []
    valPaths = []
    testPaths = []

    for i, data in enumerate(allFiles):
        if i < numTrain:
            classtype = isClass(data)
            trainPaths.append([srcPath + data, classtype])
        if numTrain <= i < (numTrain + numVal):
            classtype = isClass(data)
            valPaths.append([srcPath + data, classtype])
        if (numTrain + numVal) <= i < (numTrain + numVal + numTest):
            classtype = isClass(data)
            testPaths.append([srcPath + data, classtype])

    trainPaths = pd.DataFrame(trainPaths, columns=["Image", "Class"])
    valPaths = pd.DataFrame(valPaths, columns = ['Image','Class'])
    testPaths = pd.DataFrame(testPaths, columns = ['Image','Class'])

    return trainPaths, valPaths, testPaths


def rollingAverage(inputlist, window):
    if len(inputlist) >= window:
        while len(inputlist) > window:
            inputlist.pop(0)
    
    avg = sum(inputlist) / len(inputlist)
    return avg

def plotTrain(data, isList):
    if isList:
        data = np.array(data)

    fig, ax = plt.subplots()

    
    ax.plot(data[:,0], label = 'Train Loss')
    ax.plot(data[:,1], label = 'Val Loss')
    try:
        ax.plot(data[:,5], label = 'Unseen Losss')
    except:
        pass

    ax2 = ax.twinx()

    ax2.plot(data[:,2], label = 'Train Accuracy')
    ax2.plot(data[:,3], label = 'Val Accuracy')
    try:
        ax2.plot(data[:,4], label = 'Unseen Accuracy')
    except:
        pass

    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training Loss/Accuracy')
    plt.legend()
    plt.show()



def negateBackground(feature, image):
    coord, contour = feature
    x, y, w, h = coord

    ## create bw mask of feature
    mask = np.zeros(image.shape, dtype=np.uint8)
    output = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255,255,255), -1)

    output = np.where(mask > 0, image, mask)

    segment = output[y:(y+h),x:(x+w),:]

    ## how much of image is good content
    zeroPixels = np.sum(np.all(segment == [0,0,0], axis=-1))
    # print(segment.shape)
    totalPixels = segment.shape[0] * segment.shape[1] 
    # print(zeroPixels, totalPixels)

    percentInfo = 1 - zeroPixels / totalPixels # percent of pixels that are not black / total number of pixels


    return segment, percentInfo



def outputRegions(image, imageName, regions, imgSavePath, binaryBackground=False): ## given list of regions, segment src image per each region
    imagelist = []
    resize = (64,64) ## somewhat arbritrary; cifar10 uses 32x32, mnist uses 28x28, imagenet uses approx. 480x300. want good resolution, but not too much of upscale.
    # print(len(regions))

    # print(imgSavePath)
    for i, region in enumerate(regions):
        # if len(region) != 0:
        savePath = ''
        # segment = image[x:(x+w), y:(y+h), :]

        ## recutangular regions and contours
        if binaryBackground:
            segment, percentInfo = negateBackground(region, image)
            # print(percentInfo)
        ## if regular regions, not list of contours and rectangular regions
        else:
            x, y, w, h = region
            segment = image[y:(y+h),x:(x+w),:]
        try:
            segment = resizeImage(segment,resize)
        except:
            continue
        name = imageName + '_' + str(i)
        if imgSavePath is not None:
            # # print(segment.shape)
            # if 'HPNE' in name:
            #     savePath = imgSavePath  
            # if 'MIA' in name: 
            #     savePath = imgSavePath
            savePath = imgSavePath
            try:
                # print(name)
                # print(savePath)
                if percentInfo > 0.7: ## getting rid of images that are mostly background
                    saveImage(segment, name, savePath, secondImage=False)
                # saveImage(segment, name, savePath, secondImage=False)
            except:
                continue
        else:
            imagelist.append(segment)
        
    return imagelist
