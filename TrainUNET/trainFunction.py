from datasets import * 
from auxiliary import *
from model import UNET
import torch
import numpy as np
import os
from tqdm import tqdm



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


'''
pass a model, source datafolder, and list of hyperparameters to the function. 
the datasplit is the ratio between train and validation data

hyperparam list: learning rate, batch size, epochs, class weights. 

'''


def loadPretrainedModel(model, modelPath, keepEpoch = False):
    model.load_state_dict(torch.load(modelPath))

    return model


def saveImage(image, savePath, isOneHot):
    image = image.cpu().detach()

    ## lets use np array to save image
    image = np.array(image)
    image = image[0, :, :, :] ## remove batch dimension
    image = np.transpose(image, [1, 2, 0]) ### proper format for cv2

    # print('image shape:', image.shape)

    if isOneHot:
        image = onehotToIMG(image)
    
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # if not isOneHot:
        # image = image[0,:,:,:] ## dont know why this is necessary

    cv2.imwrite(savePath, image)
    # print('saved image to:', savePath)



def determineROI(imgA, imgB):
    ## ensure on correct device
    imgA, imgB = imgA.cpu().detach().numpy(), imgB.cpu().detach().numpy()
    
    imgA, imgB = np.array(imgA), np.array(imgB)

    ## ensure predictions are either 1 or 0
    imgA = np.where(imgA > 0.5, 1, 0)
    imgB = np.where(imgB > 0.5, 1, 0)

    #find roi
    intersection = np.logical_and(imgA, imgB)
    union = np.logical_or(imgA, imgB)
    roi = np.sum(intersection) / np.sum(union)

    return roi

## for stopping early
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


def trainFunction(model, modelSave, csvSave, trainDataPath, valDataPath, hyperparams):
    lr, batchSize, epochs, classWeights, earlyStop = hyperparams

    # miaWeight, hpneWeight = classWeights

    traindata = getDataset(trainDataPath, batchSize=batchSize, shuffle=True, pin_memory=True, eval=False)
    valdata = getDataset(valDataPath, batchSize=batchSize, shuffle=True, pin_memory=True, eval=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lossFunction = torch.nn.CrossEntropyLoss(weight=torch.tensor(classWeights).to(device))

    ## callbacks; use early stop on validation loss
    earlystopper = EarlyStopper(patience=10, min_delta=0.001)

    ## storing train data
    trainLoss = []
    valLoss = []
    trainroi = []
    valroi = []

    trainingData = []

    for epoch in range(epochs):

        epochData = []
        epochData.append(epoch)

        ## train step
        stepLoss = 0
        stepAcc = 0
        stepiter = 0

        # print(type(traindata))
        # print(len(traindata))
        # print(traindata.shape)
        for idx, batch in enumerate(tqdm(traindata)):
            x, y = batch
            x, y = x.to(device), y.to(device)

            predictions = model(x)

            predictions, y = predictions.float(), y.float()

            # print(type(predictions), type(y))
            # print(predictions.shape, y.shape)
            loss = lossFunction(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stepLoss += loss.item()

            precision, recall, f1, acc, xorimage, vals = accuracy(predictions, y)

            stepAcc += acc
            stepiter += 1
        
        etrainloss = stepLoss / stepiter
        etrainroi = stepAcc / stepiter

        tp, tn, fp, fn = vals
        epochData.append(etrainloss)
        epochData.append(etrainroi)
        epochData.append(tp)
        epochData.append(tn)
        epochData.append(fp)
        epochData.append(fn)
        # epochData.extend(etrainloss, etrainroi, tp, tn, fp, fn)
        
        # trainLoss.append(stepLoss / stepiter)
        # trainroi.append(stepAcc / stepiter)

        ## validation step

        ## random image from validation set
        randint = np.random.randint(0, len(valdata))
        stepLoss = 0
        stepAcc = 0
        stepiter = 0
        for idx, batch in enumerate(tqdm(valdata)):
            x, y, name = batch
            x, y = x.to(device), y.to(device)

            predictions = model(x)

            predictions, y = predictions.float(), y.float()

            loss = lossFunction(predictions, y)

            stepLoss += loss.item()

            precision, recall, f1, acc, xorimage, vals = accuracy(predictions, y)
            stepAcc += acc
            stepiter += 1

            if idx == randint:

                saveImage(x, valDataPath + 'Output/e' + str(epoch) + '_' + str(idx) + 'test.png', isOneHot = False)
                saveImage(y, valDataPath + 'Output/e' + str(epoch) + '_' + str(idx) + 'testmask.png', isOneHot = True)
                saveImage(predictions, valDataPath   + 'Output/e' + str(epoch) + '_' + str(idx) + 'testpred.png', isOneHot = True)
                # print(f'Image: {name[0]} | Val Loss: {loss.item()} | Val ROI: {determineROI(predictions, y)}')
        
        # valLoss.append(stepLoss / stepiter)
        # valroi.append(stepAcc / stepiter)
        evalloss = stepLoss / stepiter
        evalroi = stepAcc / stepiter

        tp, tn, fp, fn = vals
        epochData.append(evalloss)
        epochData.append(evalroi)
        epochData.append(tp)
        epochData.append(tn)
        epochData.append(fp)
        epochData.append(fn)


        ## save output data
        trainingData.append(epochData)
        npData = np.array(trainingData)
        headers = 'epoch, trainloss, trainroi, train tp, train tn, train fp, train fn, valloss, valroi, val tp, val tn, val fp, val fn'
        np.savetxt(csvSave, npData, delimiter=',', header=headers) 


        ## save model
        torch.save(model.state_dict(), modelSave)


        print(f'Epoch: {epoch} | Train Loss: {etrainloss} | Train ROI: {etrainroi} | Val Loss: {evalloss} | Val ROI: {evalroi}')

        ## early stopping
        if earlyStop:
            if earlystopper.early_stop(evalloss):
                print('Early Stopping')
                break



