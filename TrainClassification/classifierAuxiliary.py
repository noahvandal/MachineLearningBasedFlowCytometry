import os 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from classifierNets import *


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
    '''
    Dataset generator for classification network.
    The input is a pandas dataset, with images and associated labels. 

    Need: dataset split, and each image contain class ['HPNE', 'MIA'] in name.

    '''
    def __init__(self, dataframe, batchsize, numsteps, isVal = False, transformList=None):
        self.dataframe = dataframe
        self.batchsize = batchsize
        self.numsteps = numsteps
        self.Validation = isVal
        self.resize = (64,64)
        self.transformList = transformList

    
    def __len__(self):
        length = len(self.dataframe) 
        return length

    def on_epoch_end(self):
        self.dataframe = self.dataframe.reset_index(drop = True)

    def __getitem__(self, index):
        images = []
        labels = []

        img = cv2.imread(self.dataframe["Image"][index])
        img = cv2.resize(img,self.resize)

        label = np.array(self.dataframe["Class"][index])

        images.append(img)
        labels.append(label)

        if self.transformList:

            img = np.transpose(img, [2, 0, 1]) ## torch expects [:.., W, H]
            img = torch.from_numpy(img)
            img = self.transformList(img)
            img = img.float()



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
    '''
    Given path to folder, create dataframe with image path and class.
    '''
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
    '''
    Will calculate rolling average of signal desired; used for viewing while training. 
    '''
    if len(inputlist) >= window:
        while len(inputlist) > window:
            inputlist.pop(0)
    
    avg = sum(inputlist) / len(inputlist)
    return avg

def plotTrain(data, isList):
    '''
    plot train, val data while training. 
    '''
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

    '''
    Negated background to prevent artifacts from being included as part of learned features.
    '''

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
    totalPixels = segment.shape[0] * segment.shape[1] 

    percentInfo = 1 - zeroPixels / totalPixels # percent of pixels that are not black / total number of pixels


    return segment, percentInfo


def saveImage(image, imageName, savePath, secondImage=False):
    imageSavePath = savePath + imageName + '.png'

    if secondImage is not False:
        image = scaleImage(image)
        secondImage = scaleImage(secondImage)
        fImage = stackImages(image, secondImage)
        cv2.imwrite(imageSavePath, fImage)

    else:
        image = scaleImage(image)
        cv2.imwrite(imageSavePath, image)


def stackImages(img1, img2):  # both images must be input as numpy array dtype
    h, w, _ = img1.shape

    if (img2.shape[0] != h) or (img2.shape[1] != w):
        cv2.resize(img2, (h, w))

    if len(img2.shape) == 2:
        img2 = np.expand_dims(img2, axis=-1)
        img2 = np.float32(img2)

        # if is single channel grayscale, convert to rgb for concatenation and better saving.
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    outimage = np.hstack([img1, img2])

    return outimage


def scaleImage(image):
    maxPixel = np.max(image)
    if maxPixel > 1:  # if a pixel value is greater than 1, then it is not normalized and is in range 0-255 most likely
        image = image
    else:
        image = 255*image

    return image


def outputRegions(image, imageName, regions, imgSavePath, binaryBackground=False): 
    '''
    given list of regions, segment src image per each region
    '''
    imagelist = []
    resize = (64,64) ## somewhat arbritrary; cifar10 uses 32x32, mnist uses 28x28, imagenet uses approx. 480x300. want good resolution, but not too much of upscale.

    for i, region in enumerate(regions):
        savePath = ''

        ## recutangular regions and contours
        if binaryBackground:
            segment, percentInfo = negateBackground(region, image)
        else:
            x, y, w, h = region
            segment = image[y:(y+h),x:(x+w),:]
        try:
            segment = cv2.resize(segment, resize, cv2.INTER_LINEAR)
        except:
            continue
        name = imageName + '_' + str(i)
        if imgSavePath is not None:
            savePath = imgSavePath
            try:
                if percentInfo > 0.7: ## getting rid of images that are mostly background
                    saveImage(segment, name, savePath, secondImage=False)
            except:
                continue
        else:
            imagelist.append(segment)
        
    return imagelist




def whichModelToReturn(index):
    '''
    list method wasnt working for some reason, so using if statements instead.
    '''
    # if index == 0:
        # return Classifier_v1()
    if index == 1:
        return Classifier_v2()
    if index == 2:
        return Classifier_v3()
    if index == 3:
        return Classifier_v4()
    if index == 4:
        return Classifier_v5()
    if index == 5:
        return Classifier_v6()
    if index == 6:
        return Classifier_v7()
    if index == 7:
        return Classifier_v8()
    if index == 8:
        return Classifier_v9()
    if index == 9:
        return Classifier_v10()
    if index == 10:
        return Classifier_v11()
    if index == 11:
        return Classifier_v12()
    if index == 12:
        return Classifier_v13()
    if index == 13:
        return Classifier_v14()
    if index == 14:
        return Classifier_v4()  ## repeating to ensure duplicativity
    if index == 15:
        return Classifier_v15()
    if index == 16:
        return Classifier_v16()
    if index == 17:
        return Classifier_v17()
    if index == 18:
        return Classifier_v18()
    if index == 19:
        return Classifier_v19()
    # if index == 20:
    #     return Classifier_v20()
    # if index == 21:
    #     return Classifier_v21()
    # if index == 22:
    #     return Classifier_v22()
    # if index == 23:
    #     return Classifier_v23()
    # if index == 24:
    #     return Classifier_v24()
    # if index == 25:
    #     return Classifier_v25()