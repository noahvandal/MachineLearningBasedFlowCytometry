from trainFunction import trainFunction, loadPretrainedModel
from model import UNET
import torch
import numpy as np

np.random.seed(43)
torch.manual_seed(43)




if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
 

def main():
    ## hyperparameters
    lr = 1e-4
    batchSize = 16
    epochs = 100
    classWeights = [20, 1] ## since there are about 20 times more background pixels than foreground pixels
    earlyStop = True  ## patience is set to 10 by default

    hyperparams = [lr, batchSize, epochs, classWeights, earlyStop]

    ## data
    rootPath = '/home/noahvandal/my_project_dir/my_project_env/TrainUNET/'
    trainData = rootPath + 'Dataset/FineSplit/Train/'
    valData = rootPath + 'Dataset/FineSplit/Val/'

    ## model
    model = UNET(in_channels = 3, classes = 2)
    ## load pretrained model. If not, comment out this line
    model.load_state_dict(torch.load(rootPath + 'Models/051523_2c_2e_1e-3lr_16bs_100ep.pt'))
    model.to(device)

    modelName = '051523_2c_2e_1e-4lr_16bs_100ep_FineTune'
    modelSave = rootPath + 'Models/' + modelName + '.pt'

    ## csv output 
    csvSave = rootPath + 'LearningData/' + modelName + '.csv'

    trainFunction(model,modelSave, csvSave, trainData, valData, hyperparams)



if __name__ == '__main__':
    main()