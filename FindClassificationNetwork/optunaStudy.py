# %%
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from classifierAuxiliary import *
from torch.utils.data import Dataset


# Build a model by implementing define-by-run design from Optuna
def build_model_custom(trial):
    
    
    n_conv_layers = trial.suggest_int("n_layers", 1, 3)
    n_fc_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    kernellist = 0

    imageshape = 64 ## shape of input image

    in_features = 3
    
    for i in range(n_conv_layers):
        
        out_features = trial.suggest_int("n_units_conv{}".format(i), 4, 256)
        kernel = trial.suggest_int("kernel_conv{}".format(i), 1, 9, step=2)
        
        layers.append(nn.Conv2d(in_features, out_features, kernel))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))

        imageshape = int((imageshape - kernel + 1) / 2) ## caluclating new image shape, / 2 for maxpool

        print(imageshape, kernel)
        in_features = out_features
   
    # calculating inptu dimensions for fc layer
    print(imageshape, out_features)
    in_features = int(imageshape * imageshape * out_features)

    layers.append(nn.Flatten()) ## flatten input for fc layer
    
    
    for i in range(n_fc_layers):
        
        out_features = trial.suggest_int("n_units_fc{}".format(i), 4, 256)
     
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        in_features = out_features
        
    layers.append(nn.Linear(in_features, 2))
    layers.append(nn.Softmax())

    print(layers)
    
    return nn.Sequential(*layers)

# Train and evaluate the accuracy of neural network with the addition of pruning mechanism
def train_and_evaluate(param, model, trial, sourcePath):

  
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, param['optimizer'])(model.parameters(), lr= param['learning_rate'])



    classes = ['HPNE', 'MIA']

    ## for hyperparamsearch2 lr was 0.000001; for 1, was 0.00001; for v23, 24, 25 was e^-7
    lr = 0.00001  ## for all train before v12, was 0.00001; for v12, was 0.0001; for v15 was 0.00001
    # EPOCHS = 2000
    keepTrainEpoch = False
    epoch = 0

    numImages = len(os.listdir(sourcePath))

    numtrain = int(numImages * 0.6)
    numVal = int(numImages * 0.3)
    numTest = numImages - numtrain - numVal

    print(numtrain, numVal, numTest)

    trainDataSet, valDataSet, testDataSet = datasetAcquirerShufflerWithTest(sourcePath, numtrain, numVal, numTest) #train, val sizees respectively.

    # EPOCHS = 200
    epoch = 0
    
    batchsize = 16
    valbatchsize = 16
    valnumsteps = 5
    numsteps = 20
    transformList = []
    
    train = getDataset(trainDataSet,batchsize,numsteps,isVal=False, transformList=transformList)
    val = getDataset(valDataSet,valbatchsize,valnumsteps,isVal=True)

    print('dataset acquired')


    for e in range(epoch, EPOCHS):
        tLoss = 0
        vLoss = 0
        tAcc = 0
        vAcc = 0
        acc = 0


        trainstep = 0
        valStep = 0

        for stepsize in range(numsteps):
            # print('train step: ',stepsize)
            for i, data in enumerate(train):
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
                # print(acc)
                tAcc += acc
                trainstep += 1
        
        tAcc = (tAcc / (trainstep)) / numsteps

        tLoss = tLoss.cpu().detach().numpy()
        tLoss = (tLoss / trainstep) 



        valstep = 0
        with torch.no_grad(): ## no need to calculate gradients for validation
            for stepsize in range(valnumsteps):
                # print('val step: ',stepsize)
                for i, data in enumerate(val):
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
                    # print(acc)
                    vAcc += acc
                    valstep += 1
        
        # print(vAcc, valstep, valnumsteps)
        vAcc = (vAcc / (valstep)) 

        print('Epoch:',e, 'vAcc:',vAcc)


        # Add prune mechanism
        trial.report(vAcc, e)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return vAcc
  
# Define a set of hyperparameter values, build the model, train the model, and evaluate the accuracy
def objective(trial):

    start = time.time()

    rootPath = ''
    sourcePath = rootPath + 'Dataset/Source/EntireDataset/CleanOutput/'

     
    params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1e-2),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
            }

    model = build_model_custom(trial)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
    print('model created!')
    accuracy = train_and_evaluate(params, model, trial, sourcePath)


    end = time.time()
    clear_output(wait=True)
    print('time taken: ', end - start)

    return accuracy
  
  
EPOCHS = 2

# sampler = optuna.samplers.QMCSampler()
sampler = optuna.samplers.TPESampler(n_startup_trials=10)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100)

# %% 


best_trial = study.best_trial

for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))
# %%


optuna.visualization.plot_intermediate_values(study)
# %%

optuna.visualization.plot_optimization_history(study)
# %%
