import torch
import cv2
import numpy as np
from datasets import *
from auxiliary import *
import random
from IPython.display import clear_output


def showImage(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def onehotToBW(image,isTensor=False, outputAsRGB=False):   
    '''
    Same function as in 'datasets' file, but this one is for testing
    '''
    if isTensor:
        image = convertTensorTypeToNumpy(image)

    output = np.zeros([image.shape[0], image.shape[1]])
    output = np.expand_dims(output, axis=-1)
    image = np.argmax(image, axis=-1, keepdims=True)
    output[np.all(image == 1, axis=-1)] = 255 ## background 
    output[np.all(image == 0, axis=-1)] = 0 ## foreground
    output = output.astype('uint8')

    if outputAsRGB:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    return output


def compareTwoImages(A, B, C):
    output = np.hstack([A, B, C])

    return output


def convertTensorTypeToNumpy(image):
    '''
    Necessary for unloading model output
    '''
    image = image.cpu().detach().numpy()
    image = image[0,:,:,:] # getting rid of the batch dimension
    image = np.transpose(image, [1,2,0])

    return image
    


def testFunction(model, sourceDataset,imageSavePath, csvSavePath, device):

    numClasses = 2

    testBatch = 1
    testSteps = 1
 
    valData = getDataset(sourceDataset, testBatch, testSteps, numClasses, True, True, False)
    print('Test data acquired')

    inferList = []

  
    for step in range(testSteps):
        for i, (input, target, name) in enumerate(valData):

            origimg = input
            input, target = input.to(device), target.to(device)

            output = model(input)


            precision, recall, F1, acc, xorImg, tpVals = accuracy(output, target)

            iou = IoU(output, target)

            ## convert to image class as numpy array
            output = onehotToBW(output, True, True)

            origimg = convertTensorTypeToNumpy(origimg)

            ## prform some modifications to images to make them more readable
            origimg = 255 * origimg
            xorImg = np.transpose(xorImg, [1,2,0]).astype('uint8')
            xorImg = 255 * xorImg
            xorimg = cv2.cvtColor(xorImg, cv2.COLOR_GRAY2RGB)

            ## create the concatenated image
            saveImage = compareTwoImages(origimg, output, xorimg)


            name = os.path.basename(str(name)) ## getting only filename
            name = name[:-3] ## removing the **',)**

            ## save the concatenated image
            finalSave = imageSavePath + name
            cv2.imwrite(finalSave, saveImage)
            
            ## save data regarding test inference
            inferList.append([name, acc, F1, precision, recall, iou[0], iou[1]])

            ## print out number of images processed
            clear_output(wait=True)
            print(i)
            print(iou)
   
    


    np.savetxt(csvSavePath, inferList, delimiter=",", header = 'Name, Accuracy,\
                F1, Precision, Recall, iou cell, iou background',fmt='%s')



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('Using GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')


    # print('test')
    from model import UNET

    modelPath = 'Models/041123_2c_v3_wAugs_p10.pt'
    imagePath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/UNET_MC_PyTorch/FineTuneMarchModel/PureFineTuneTrain/'
    imageSavePath = 'Test/'
    csvSavePath = '041123_2c_v3_wAugs_p10.csv'

    model = UNET().to(device)
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    testFunction(model, imagePath, imageSavePath, csvSavePath, device)

    print('done')

