## this script runs the classification portion of the model. 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import random
from IPython.display import clear_output
from classifierAuxiliary import *
from hyperparamSearch import *


if torch.cuda.is_available():
    device = 'cuda:1'
else:
    device = 'cpu'


torch.manual_seed(43)
np.random.seed(43)
random.seed(43)




def testFunction(testPath,classes, modelPath, csvSave):
    '''
    Perform inference on the test dataset. 
    '''
    testlist = os.listdir(testPath)

    outputList = []
    runningAvgAccCount = 0
    runningAvgAcc = 0

    model = ClassifierHyperparam_v3().to(device) ## batch size of 1
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model Successfully Loaded')

    for i, test in enumerate(testlist):
        label = np.array(isClass(test))
        img = cv2.imread(testPath + test)
        img = cv2.resize(img, (64,64))
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, 0)

        label = np.expand_dims(label,0)


        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        img = img.float()
        label = label.float()
        
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
        
        _, index = torch.max(outputs, dim=1)
        _,labelIndex = torch.max(label, dim=1) 

        ## see whether each image was correct or not. 
        acc, comparelist = checkAccuracy(index,labelIndex,classes, printOutput = False)

        ## get running average of the accuracy
        runningAvgAccCount += acc
        runningAvgAcc = runningAvgAccCount / (i + 1)

        print(len(comparelist), comparelist)
        outputList.append([test,acc,comparelist[0][0], comparelist[0][1], runningAvgAcc])

        if i%20 == 0:
            print(i)
        
    print(type(outputList), len(outputList))
    outputList = np.array(outputList)
    np.savetxt(csvSave, outputList, delimiter=',',header='',fmt='%s')
    print('All images tested')



def testFunctionFromDataFrame(model, dataFrame,classes, modelPath, csvSave):
    '''
    inputting a pandas dataframe instead of loading data from a folder.
    '''

    outputList = []
    runningAvgAccCount = 0
    runningAvgAcc = 0

    model = model.to(device)
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model Successfully Loaded')


    dataSet = getDataset(dataFrame, 1, 1, isVal=False)

    for i, data in enumerate(dataSet):
        img, label, name = data
        name = os.path.basename(name[0])
        
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
        
        _, index = torch.max(outputs, dim=1)
        _,labelIndex = torch.max(label, dim=1) 

        ## see whether each image was correct or not. 
        acc, comparelist = checkAccuracy(index,labelIndex,classes, printOutput = False)

        ## get running average of the accuracy
        runningAvgAccCount += acc
        runningAvgAcc = runningAvgAccCount / (i + 1)

        outputList.append([name,acc,comparelist[0][0], comparelist[0][1], runningAvgAcc])

        if i%20 == 0:
            print(i)
        
    print(type(outputList), len(outputList))
    print(outputList)
    outputList = np.array(outputList)
    np.savetxt(csvSave, outputList, delimiter=',',header='',fmt='%s')
    print('All images tested')



class EarlyStopper:
    '''
    Early stopping on the training loop.
    '''
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def trainFunction(model, trainPath, valPath, sourcePath, unseenFolder, saveModelPath, loadModelPath, csvSave, fileUsedNames, loadFromDataframe, loadModel=False, currentModel = 1, numModels=1):
    batchsize = 16

    classes = ['HPNE', 'MIA']

    ## for hyperparamsearch2 lr was 0.000001; for 1, was 0.00001; for v23, 24, 25 was e^-7
    # lr = 0.0000001  ## for all train before v12, was 0.00001; for v12, was 0.0001; for v15 was 0.00001
    lr = 0.00003 ## found using hyperparam tune from optuna
    EPOCHS = 2000
    keepTrainEpoch = False
    epoch = 0

    numImages = len(os.listdir(sourcePath))

    #find number of mia, hpne in datset and adjust accordingly
    numMia = np.sum([1 for i in os.listdir(sourcePath) if 'MIA' in i])
    numHpne = np.sum([1 for i in os.listdir(sourcePath) if 'HPNE' in i])

    print(numImages, numMia, numHpne)

    miaWeight = 1 - numHpne / (numMia + numHpne)
    hpneWeight = 1 - numMia / (numMia + numHpne)

    print('MIA weight', miaWeight)
    print('HPNE weight', hpneWeight)


    rollingAvgTrainAcc = []
    rollingAveragValAcc = []


    lossWeights = [hpneWeight,miaWeight] ### inherent class imbalance, 16 HPNE per 10 MIA, trying to remedy; https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(lossWeights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr, betas=(0.9,0.999), weight_decay=0.0001)


    earlystop = EarlyStopper(patience=10, min_delta=0.0001)

    
    ## getting number of train and val images; we want a 6:3:1 train:val:test split.
    numtrain = int(numImages * 0.6)
    numVal = int(numImages * 0.3)
    numTest = numImages - numtrain - numVal

    print(numtrain, numVal, numTest)
    
    batchsize = 16
    numsteps = 1
    valbatchsize = 4
    valnumsteps = 1

    if loadFromDataframe:
        trainDataSet = pd.read_pickle(fileUsedNames + 'BinaryBackMay2_trainSet.pkl')
        valDataSet = pd.read_pickle(fileUsedNames + 'BinaryBackMay2_valSet.pkl')
        testDataSet = pd.read_pickle(fileUsedNames + 'BinaryBackMay2_testSet.pkl')
    
    else:
        trainDataSet, valDataSet, testDataSet = datasetAcquirerShufflerWithTest(sourcePath, numtrain, numVal, numTest) #train, val sizees respectively.

        unseenSet = createDataFrameFolder(unseenFolder)
        
        #save the train, val, and test sets for later use.
        trainDataSet.to_pickle(fileUsedNames + 'BinaryBackMay2_trainSet.pkl')
        valDataSet.to_pickle(fileUsedNames + 'BinaryBackMay2_valSet.pkl')
        testDataSet.to_pickle(fileUsedNames + 'BinaryBackMay2_testSet.pkl')


    ## for data augmentation; not used in this version
    transformList = transforms.Compose([
        # transforms.RandomHorizontalFlip(0.1),
        # transforms.RandomVerticalFlip(0.1),
        # transforms.RandomRotation(45),
        # transforms.RandomInvert(0.1),
        # transforms.RandomEqualize(0.1),
        # transforms.RandomCrop(64, padding=4, padding_mode='reflect'), ## added in v12 due to paper resarch
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  ## added 4/03/23 to keep training on morphology and not color.
    ])

    hpnecount = 0
    miacount = 0

    for row in trainDataSet['Image']:
        if 'HPNE' in row:
            hpnecount += 1
        if 'MIA' in row:
            miacount += 1
    print('HPNE count', hpnecount)
    print('MIA Count', miacount)
            
    
    ## load data
    trainSet = getDataset(trainDataSet,batchsize,numsteps,isVal=False, transformList=transformList)
    valSet = getDataset(valDataSet,valbatchsize,valnumsteps,isVal=True)

    unseenImgs = getDataset(unseenSet,1,1,isVal=True)

    print(trainSet)
    trainValues = []

    RAaccTrain = []
    RAaccVal = []
    RAlossTrain = []
    RAlossVal = []
    

    print(trainSet.__sizeof__(), valSet.__sizeof__())
    print(len(trainSet), len(valSet))
    
    if loadModel:
        checkpoint = torch.load(loadModelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if keepTrainEpoch:
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            epoch = checkpoint['epoch'] + 1
            trainValues = checkpoint['loss_values']
        print('Model Successfully Loaded')

    startTime = time.time()
    
    for e in range(epoch, EPOCHS):
        start = time.time()
        tLoss = 0
        vLoss = 0
        tAcc = 0
        vAcc = 0
        acc = 0
        uloss = 0
        uacc = 0


        trainstep = 0

        for stepsize in range(numsteps):
            for i, data in enumerate(trainSet):
                inputs, label, name = data

                optimizer.zero_grad()

                inputs = inputs.to(device)
                label = label.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, label)
                tLoss += loss   
                loss.backward()
                optimizer.step() ## decay if necessary for adam

                outLoss, index = torch.max(outputs, dim=1)
                _,labelIndex = torch.max(label, dim=1)

                ## get accuracy metrics
                acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)
                tAcc += acc
                trainstep += 1
        
        tAcc = (tAcc / (trainstep)) 

        tLoss = tLoss.cpu().detach().numpy()
        tLoss = (tLoss / trainstep) 

        valstep = 0
        with torch.no_grad(): ## no need to calculate gradients for validation
            for stepsize in range(valnumsteps):
                for i, data in enumerate(valSet):
                    # print(i)
                    inputs, label, index = data

                    inputs = inputs.to(device)
                    label = label.to(device)

                    outputs = model(inputs)

                    vLoss += criterion(outputs, label)
                    outLoss, index = torch.max(outputs, dim=1)
                    _,labelIndex = torch.max(label, dim=1)

                    ## get accuracy metrics
                    acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)
                    vAcc += acc
                    valstep += 1
                
        ## this step was used on a third dataset; not really all that useful in the long run but kept for posterity
        unseenstep = 0
        with torch.no_grad(): ## no need to calculate gradients for validation
            for i, data in enumerate(unseenImgs):
                inputs, label, index = data
                inputs, label = inputs.to(device), label.to(device)

                outputs = model(inputs)

                uloss += criterion(outputs, label)
                outLoss, index = torch.max(outputs, dim=1)
                _,labelIndex = torch.max(label, dim=1)
                acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)

                uacc += acc
                unseenstep += 1
        
        vAcc = (vAcc / (valstep)) 

        vLoss = vLoss.cpu().detach().numpy()
        vLoss = (vLoss / valstep) 

        uacc = (uacc / (unseenstep)) 
        uloss = uloss.cpu().detach().numpy()
        uloss = (uloss / unseenstep)


        RAaccTrain.append(tAcc)
        RAaccVal.append(vAcc)
        RAlossTrain.append(tLoss)
        RAlossVal.append(vLoss)

        clear_output(wait=True)

        print(tLoss, vLoss, tAcc, vAcc, uacc, uloss)
        trainValues.append([tLoss,vLoss,tAcc, vAcc, uacc, uloss])

        raTa = rollingAverage(RAaccTrain, 10)
        raVa = rollingAverage(RAaccVal, 10)
        raTl = rollingAverage(RAlossTrain, 100)
        raVl = rollingAverage(RAlossVal, 100)


        ## terminate sequence if difference between val and train accuracy is too large, after at least 10 epochs
        if abs(raVa - raTa) > 0.1 and e > 10: 
            print('Terminating due to large difference between train and val accuracy')
            break


        plotTrain(trainValues, True)

        print('One epoch down! here is the loss:', tLoss, vLoss)
        print('Here is the RA loss:', raTl, raVl)
        print('Here is the accuracy:', tAcc, vAcc)
        print('Here is the RA accuracy:', raTa, raVa)

        print('Epoch number: ', e)

        
        ## saving model output
        torch.save({
            'model_state_dict': model.state_dict(),  ## these are the weights and overall configuration
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': trainValues
        }, saveModelPath)

        if e % 1 == 0:
            header = 'Train Loss, Val Loss, Train Accuracy, Val Accuracy, Unseen Accuracy, Unseen Loss'
            np.savetxt(csvSave, trainValues, delimiter=',',header=header)

        ## early stopping
        if earlystop.early_stop(vLoss):
            print("Early stopping")
            break

        end = time.time()

        print('Time for epoch (s):', end - start)
        estTimeInSec = ((end - startTime)/ (e + 1) * (EPOCHS - e + 1))
        m = estTimeInSec // 60
        s = estTimeInSec % 60
        print(f'Time left:{m:.2f} m, {s:.2f} s')
        totalTimeLeft = (((end - startTime)/ (e + 1)) * EPOCHS * (numModels - currentModel))
        h, remainder = divmod(totalTimeLeft, 3600)
        m, s = divmod(remainder, 60)
        print(f'Total time left:{h:.2f} h, {m:.2f} m, {s:.2f} s')


    return testDataSet ##if test function inputs a dataframe for testing




if __name__ == '__main__':
    rootPath = ''


    model = ClassifierHyperparam_v3()
    model.to(device)

    print('device used:', device)
    print(model.type)

    trainPath = rootPath + 'Dataset/Train/'
    valPath = rootPath + 'Dataset/Val/'
    testPath = rootPath + 'Dataset/Source/NewSource/AllTestTogether/'

    ## the root image directory; the software will perform the train / val split itself
    sourcePath = rootPath + 'Dataset/Source/MajorityDataset/CleanOutput/'
    ## optional unseed dataset
    unseenPath = rootPath + 'Dataset/Source/UnseenImgs/Output/'

    ## where to load the pickled pandas dataset from
    filesUsedSave = rootPath + 'Dataset/Source/AllTrainVideos/'
    loadFromDataframe = False

    ## name, and where to save output model
    modelSrc = rootPath + 'Dataset/HyperparamSearch3/Model/'
    modelName = '052023_2c_Hyperparam_v3_fillTrain_noAug_drop10_bn_binaryBack_adamW_631split_earlyTerminate_invweight_MajorityDatset_drop5'
    saveModelPath = modelSrc + modelName +'.pt'

    ## if loading a model from a previous run, specify the path here
    loadModelPath = modelSrc + '040623_2c_v14_pureTest_Linux_nodrop_noAugsSingleFolder_t2.pt'

    ## where to save the csv data
    csvSave = rootPath + 'Dataset/HyperparamSearch3/LearningData/' + modelName + '.csv'
    csvSaveTest = rootPath + 'Dataset/HyperparamSearch3/LearningData/' + modelName  + 'ALLTRAIN_TestData.csv'

    classes = ['HPNE','MIA']

    loadModel = False

    if not os.path.exists(modelSrc):
        os.makedirs(modelSrc)
    
    ## will return the test dataframe
    currentModel = 25
    testSet = trainFunction(model, trainPath,valPath, sourcePath, unseenPath, saveModelPath, loadModelPath, csvSave, filesUsedSave,loadFromDataframe, loadModel, currentModel, 26)

    sourcePath = rootPath + 'Dataset/Source/UnseenImgs/Output/'
    csvSave = rootPath + 'Dataset/Source/UnseenImgs/output16may.csv'
    # sourcePath = rootPath + 'Dataset/Source/AllTrainVideos/Output/'
    # csvSave = rootPath + 'Dataset/Source/AllTrainVideos/output16may.csv'
    testFunction(sourcePath,classes,saveModelPath,csvSave)

    testFunctionFromDataFrame(model, testSet,classes,saveModelPath,csvSaveTest)