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
# from classifierNets import Classifier_v19
from classifierAuxiliary import *
# from hyperparamSearch import Classifier_v1, Classifier_v2, Classifier_v3, Classifier_v4, Classifier_v5
# import hyperparamSearch
from hyperparamSearch import *

if torch.cuda.is_available():
    device = 'cuda:1'
else:
    device = 'cpu'


torch.manual_seed(43)
np.random.seed(43)
random.seed(43)




def testFunction(testPath,classes, modelPath, csvSave):
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
        
        # print(outputs, label)
        _, index = torch.max(outputs, dim=1)
        _,labelIndex = torch.max(label, dim=1) 
        # print('train inaccuracy')
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

    outputList = []
    runningAvgAccCount = 0
    runningAvgAcc = 0

    model = model.to(device)
    # model = Classifier_v19().to(device) ## batch size of 1
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model Successfully Loaded')


    dataSet = getDataset(dataFrame, 1, 1, isVal=False)

    for i, data in enumerate(dataSet):
        # label = np.array(isClass(test))
        img, label, name = data
        name = os.path.basename(name[0])
        
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
        
        # print(outputs, label)
        _, index = torch.max(outputs, dim=1)
        _,labelIndex = torch.max(label, dim=1) 
        # print('train inaccuracy')
        ## see whether each image was correct or not. 
        acc, comparelist = checkAccuracy(index,labelIndex,classes, printOutput = False)

        ## get running average of the accuracy
        runningAvgAccCount += acc
        runningAvgAcc = runningAvgAccCount / (i + 1)

        # print(len(comparelist), comparelist)
        outputList.append([name,acc,comparelist[0][0], comparelist[0][1], runningAvgAcc])

        if i%20 == 0:
            print(i)
        
    print(type(outputList), len(outputList))
    print(outputList)
    outputList = np.array(outputList)
    np.savetxt(csvSave, outputList, delimiter=',',header='',fmt='%s')
    print('All images tested')

class EarlyStopper:
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
    # model = Classify(batchsize).to(device)
    # model = Classifier_v12()
    # model = Classifier_v19()
    # model.to(device)
    # loadModel = True

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

    # miaWeight = 1
    # hpneWeight = 1  
    print('MIA weight', miaWeight)
    print('HPNE weight', hpneWeight)


    rollingAvgTrainAcc = []
    rollingAveragValAcc = []


    # lossWeights = [7, 5] ## gives more weight to the MIA class.
    lossWeights = [hpneWeight,miaWeight] ### inherent class imbalance, 16 HPNE per 10 MIA, trying to remedy; https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(lossWeights).to(device))
    # optimizer = optim.Adam(model.parameters(), lr, betas=(0.9,0.999), weight_decay=0.00001)
    optimizer = optim.AdamW(model.parameters(), lr, betas=(0.9,0.999), weight_decay=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    # optimizer = optim.Adagrad(model.parameters(), lr)
    # optimizer = optim.NAdam(model.parameters(), lr)

    earlystop = EarlyStopper(patience=10, min_delta=0.0001)
    # callbacks = [early(monitor='val_loss', patience=75, verbose=1, mode='min', restore_best_weights=True)]

    
    ## getting number of train and val images; we want a 8:1:1 train:val:test split.
    # numImages = len(os.listdir(sourcePath))

    ## on 5/1/23 changes split to 6/3/1 train/val/test; prior to this was an 8/1/1 split
    
    numtrain = int(numImages * 0.6)
    numVal = int(numImages * 0.3)
    numTest = numImages - numtrain - numVal

    print(numtrain, numVal, numTest)
    
    batchsize = 16
    # numsteps = int(numtrain / batchsize)
    numsteps = 1
    valbatchsize = 4
    valnumsteps = 1

    ## initialize datasets first just for delcaration of variables and in case shuffle doesnt work. Otherwise will use shuffle dataset. 
    
    # trainDataSet = createDataset(trainPath)
    # valDataSet = createDataset(valPath)

    # trainSet = getDataset(trainDataSet,batchsize,numsteps,isVal=False)

    # trainDataSet, valDataSet = datasetAcquirerShuffler(sourcePath, numtrain, numVal) #train, val sizees respectively. 

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
    
        # np.savetxt(fileUsedNames + 'trainSet.txt',trainDataSet,fmt='%s')
        # np.savetxt(fileUsedNames + 'valSet.txt',valDataSet,fmt='%s')
        # np.savetxt(fileUsedNames + 'testSet.txt',testDataSet,fmt='%s')



    
    maxValAcc = 0
    numNotMaxCount = 0

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
        # print(row)
        if 'HPNE' in row:
            hpnecount += 1
        if 'MIA' in row:
            miacount += 1
    print('HPNE count', hpnecount)
    print('MIA Count', miacount)
            
    
    # print(trainDataSet)

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

        # if e % 10 == 0:
        #     trainDataSet, valDataSet = datasetAcquirerShuffler(sourcePath, 4800, 600) #train, val sizees respectively. 
        #     trainSet = getDataset(trainDataSet,batchsize,numsteps,isVal=False)
        #     valSet = getDataset(valDataSet,valbatchsize,valnumsteps,isVal=True)

        # if e % 1000 == 0:  
            # lr = lr * 0.95 ## reduce lr
            # optimizer = optim.Adam(model.parameters(), lr, betas=(0.9,0.999))


        trainstep = 0
        valStep = 0

        for stepsize in range(numsteps):
            # print(stepsize)
            for i, data in enumerate(trainSet):
                # print(i)
                inputs, label, name = data

                # print(name)

                optimizer.zero_grad()

                # with torch.set_grad_enabled(True):

                inputs = inputs.to(device)
                label = label.to(device)

                outputs = model(inputs)
                # outputs = outputs.detach().cpu()

                # print(outputs, label)
                loss = criterion(outputs, label)
                tLoss += loss   
                loss.backward()
                optimizer.step() ## decay if necessary for adam
                outLoss, index = torch.max(outputs, dim=1)
                _,labelIndex = torch.max(label, dim=1)
                # print('train inaccuracy')
                acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)
                # print(acc)
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
                    # outputs = outputs.detach().cpu()

                    # outputs = nn.Softmax(dim=0)(outputs)
                    vLoss += criterion(outputs, label)
                    # outLoss = sm(outputs)
                    outLoss, index = torch.max(outputs, dim=1)
                    _,labelIndex = torch.max(label, dim=1)
                    # print(index, labelIndex)
                    acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)
                    # print(acc)
                    vAcc += acc
                    valstep += 1
                
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
        # tAcc = tAcc.detach().numpy()
        # vAcc = vAcc.detach().numpy()
        # print(tAcc, batchsize, vAcc, valbatchsize)

        uacc = (uacc / (unseenstep)) 
        uloss = uloss.cpu().detach().numpy()
        uloss = (uloss / unseenstep)


        RAaccTrain.append(tAcc)
        RAaccVal.append(vAcc)
        RAlossTrain.append(tLoss)
        RAlossVal.append(vLoss)

        
        # tAcc = tAcc / batchsize
        # vAcc = vAcc / valbatchsize

        # print(tAcc, batchsize, vAcc, valbatchsize)
        clear_output(wait=True)

        # print(trainstep, valstep)
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

        ## inserting this section to reduce lr if val acc is not increasing for 5 epochs
        # if e > 100:
        #     if raVa >= maxValAcc:
        #         maxValAcc = raVa
        #     if raVa < maxValAcc: 
        #         numNotMaxCount += 1
        #     if numNotMaxCount >= 5:
        #         maxValAcc = raVa
        #         lr = lr * 0.95
        #         optimizer = optim.Adam(model.parameters(), lr, betas=(0.9,0.999))
        #         numNotMaxCount = 0

        ## for k-fold cross validation, get new dataset every 50 epochs
        # if e % 50 == 0 and e != 0:
            # trainDataSet, valDataSet = datasetAcquirerShuffler(sourcePath, numtrain, numVal) #train, val sizes respectively. 
            # trainSet = getDataset(trainDataSet,batchsize,numsteps,isVal=False, transformList=transformList)
            # valSet = getDataset(valDataSet,valbatchsize,valnumsteps,isVal=True)
            
        ## reduce lr every 100 epochs:
        # if e % 100 == 0 and e != 0:
            # lr = lr * 0.57
            # optimizer = optim.Adam(model.parameters(), lr, betas=(0.9,0.999))

        # clear_output(wait=True)

        plotTrain(trainValues, True)

        print('One epoch down! here is the loss:', tLoss, vLoss)
        print('Here is the RA loss:', raTl, raVl)
        print('Here is the accuracy:', tAcc, vAcc)
        print('Here is the RA accuracy:', raTa, raVa)

        print('Epoch number: ', e)
        # for loss in lossVals:
            # print(loss)
        
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
        # print(totalTimeLeft, numModels, currentModel)
        h, remainder = divmod(totalTimeLeft, 3600)
        m, s = divmod(remainder, 60)
        print(f'Total time left:{h:.2f} h, {m:.2f} m, {s:.2f} s')


    return testDataSet ##if test function inputs a dataframe for testing

# from hyperparamSearch import Classifier_v1, Classifier_v2, Classifier_v3, Classifier_v4, Classifier_v5, Classifier_v6, Classifier_v7, Classifier_v8, Classifier_v9, Classifier_v10
# from hyperparamSearch import Classifier_v11, Classifier_v12, Classifier_v13, Classifier_v14

def whichModelToReturn(index):
    if index == 0:
        return Classifier_v1()
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
    if index == 20:
        return Classifier_v20()
    if index == 21:
        return Classifier_v21()
    if index == 22:
        return Classifier_v22()
    if index == 23:
        return Classifier_v23()
    if index == 24:
        return Classifier_v24()
    if index == 25:
        return Classifier_v25()


if __name__ == '__main__':
    rootPath = ''

    # model = Classifier_v19()
    # from hyperparamSearch import Classifier_v1, Classifier_v2, Classifier_v3, Classifier_v4, Classifier_v5


    ## the train and val paths are loaded, but not used as the source folder is where training is from
    numModels = 26
    # for i in range(23, numModels):  ## train each model for 400 epochs
    # model = whichModelToReturn(23)
    model = ClassifierHyperparam_v3()
    model.to(device)

    print('device used:', device)
    # model = Classifier_v1().to(device)
    print(model.type)
    trainPath = rootPath + 'Dataset/Train/'
    valPath = rootPath + 'Dataset/Val/'
    testPath = rootPath + 'Dataset/Source/NewSource/AllTestTogether/'

    ## the root image directory; the software will perform the train / val split itself
    sourcePath = rootPath + 'Dataset/Source/MajorityDataset/CleanOutput/'

    unseenPath = rootPath + 'Dataset/Source/UnseenImgs/Output/'

    ## where to load the pickled pandas dataset from
    filesUsedSave = rootPath + 'Dataset/Source/AllTrainVideos/'
    loadFromDataframe = False

    ## name, and where to save output model
    modelSrc = rootPath + 'Dataset/HyperparamSearch3/Model/'
    # modelName = '050623_2c_' + str(i) + 'fineLR_FullTrain_noAug_binaryback_adamW_631split'
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
    testSet = trainFunction(model, trainPath,valPath, sourcePath, unseenPath, saveModelPath, loadModelPath, csvSave, filesUsedSave,loadFromDataframe, loadModel, currentModel, numModels)

    sourcePath = rootPath + 'Dataset/Source/UnseenImgs/Output/'
    csvSave = rootPath + 'Dataset/Source/UnseenImgs/output16may.csv'
    # sourcePath = rootPath + 'Dataset/Source/AllTrainVideos/Output/'
    # csvSave = rootPath + 'Dataset/Source/AllTrainVideos/output16may.csv'
    testFunction(sourcePath,classes,saveModelPath,csvSave)

    testFunctionFromDataFrame(model, testSet,classes,saveModelPath,csvSaveTest)