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

    if isOneHot:
        image = onehotToIMG(image)
    
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imwrite(savePath, image)



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

## for stopping early; necesssary to prevent overfitting
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

    ## data 
    traindata = getDataset(trainDataPath, batchSize=batchSize, shuffle=True, pin_memory=True, eval=False)
    valdata = getDataset(valDataPath, batchSize=batchSize, shuffle=True, pin_memory=True, eval=True)

    ## optimizer, loss function type
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lossFunction = torch.nn.CrossEntropyLoss(weight=torch.tensor(classWeights).to(device))

    ## callbacks; use early stop on validation loss
    earlystopper = EarlyStopper(patience=10, min_delta=0.001)

    ## storing train data
    trainingData = []

    for epoch in range(epochs):

        epochData = []
        epochData.append(epoch)

        stepLoss = 0
        stepAcc = 0
        stepiter = 0


        for idx, batch in enumerate(tqdm(traindata)):
            x, y = batch
            x, y = x.to(device), y.to(device)

            predictions = model(x)

            predictions, y = predictions.float(), y.float()

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


        ## saving data for later use and viewing
        tp, tn, fp, fn = vals
        epochData.append(etrainloss)
        epochData.append(etrainroi)
        epochData.append(tp)
        epochData.append(tn)
        epochData.append(fp)
        epochData.append(fn)

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
                ## saving one image for viewing from the validation set; this way we can see how the training progressed. 
                saveImage(x, valDataPath + 'Output/e' + str(epoch) + '_' + str(idx) + 'test.png', isOneHot = False)
                saveImage(y, valDataPath + 'Output/e' + str(epoch) + '_' + str(idx) + 'testmask.png', isOneHot = True)
                saveImage(predictions, valDataPath   + 'Output/e' + str(epoch) + '_' + str(idx) + 'testpred.png', isOneHot = True)

        evalloss = stepLoss / stepiter
        evalroi = stepAcc / stepiter

        ## saving data for later use and viewing
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



