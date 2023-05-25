import torch
from collections import namedtuple
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# create labels for each of the classes, resepective to the color scheme given for mask creation
Label = namedtuple('Label', ['name', 'id', 'color'])
# labels = [Label('HPNE', 0, (0, 255, 255)),
#           Label('MIA', 1, (255, 0, 255)),
#           Label('PSBead', 2, (255, 255, 0)),
#           Label('Background', 3, (255, 255, 255))]

labels = [Label('HPNE', 0, (0, 255, 255)),
          Label('MIA', 1, (255, 0, 255)),
          Label('PSBead', 2, (255, 255, 0)),
          Label('Background', 3, (255, 255, 255))]


cellLabels = [Label('HPNE', 0, (0, 255, 255)),
              Label('MIA', 1, (255, 0, 255)),
              #   Label('PSBead', 2, (255, 255, 0)),
              Label('Background', 2, (255, 255, 255))]

name2label = {label.name: label for label in labels}
id2label = {label.id: label for label in labels}
color2label = {label.color: label for label in labels}
cellColor2Label = {label.color: label for label in cellLabels}

# return dataset when called upon


class Dataset(Dataset):
    def __init__(self, rootDir, transform=None, eval=False):
        self.transform = transform
        self.maskList = []
        self.imgList = []
        self.eval = eval

        # self.maskPath = os.path.join(os.getcwd(), rootDir + '/Images/')
        # self.imgPath = os.path.join(os.getcwd(), rootDir + '/Masks/')
        self.maskPath = os.path.join(rootDir + '/Masks/')
        self.imgPath = os.path.join(rootDir + '/Images/')

        # print('maskpath', self.maskPath, self.imgPath)

        ## since mask and image are the same name, we can just use one of them
        imgItems = os.listdir(self.imgPath)
        # print(imgItems)

        maskItems = [rootDir + '/Masks/' + path for path in imgItems]
        imgItems = [rootDir + '/Images/' + path for path in imgItems]

        self.maskList.extend(maskItems)
        self.imgList.extend(imgItems)
        # print(self.maskList)
        # print(self.imgList)

    def __len__(self):
        length = len(self.imgList)
        return length

    def __getitem__(self, index):
        imgPath = self.imgList[index]
        maskPath = self.maskList[index]

        img = Image.open(imgPath)
        mask = Image.open(maskPath)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = transforms.ToTensor()(img)
        # mask = transforms.ToTensor()(mask)
        # mask = np.array(mask)
        # print(mask.shape)
        mask = np.transpose(mask, [2, 0, 1])
        mask = rgbToOnehotBW(mask, cellColor2Label)
        # print('mask shape', mask.shape)

        # mask = mask[2, :, :]  # removing the alpha channel (not really needed)
        img = img.type(torch.FloatTensor)
        mask = torch.from_numpy(mask)
        # mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)

        # print(type(img), type(mask))
        # print(img.shape, mask.shape)

        if self.eval:
            return img, mask, self.imgList[index]
        else:
            return img, mask

        return img, mask

# getting datset using dataload function and native functions from torch


def getDataset(rootDir, transform=None, batchSize=1, shuffle=True, pin_memory=True, eval=False):
    data = Dataset(rootDir=rootDir, transform=transform, eval=eval)
    dataLoaded = torch.utils.data.DataLoader(data, batch_size=batchSize,
                                             shuffle=shuffle, pin_memory=pin_memory)

    return dataLoaded

# saving images when called upon


def saveImages(tensor_pred, folder, image_name):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}\{image_name}.png"
    tensor_pred.save(filename)

# function that is called when training model


def trainFunction(data, model, optimizer, lossFunction, device):
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)

        predictions = model(X)

        loss = lossFunction(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()



def rgbToOnehotNew(rgb, colorDict):
    rgb = np.array(rgb).astype('int32')
    dense = np.zeros(rgb.shape[:2])
    for label, color in enumerate(colorDict.keys()):
        color = np.array(color)
        pixel = np.zeros(len(colorDict.keys()))
        pixel[label] = 1 ## converting to one-hot encoding instead of sparse.

        ## for each class present, match the color and set the pixel to the class label
        if label < len(colorDict.keys()):
            dense[np.all(rgb == color, axis=-1)] = pixel

    return dense


'''
wherever there is a cell, make it black with encoding position of 0; 
wherever there is no cell, make it white with encoding position of 1.
'''
def rgbToOnehotBW(rgb, colorDict):
    rgb = np.array(rgb).astype('int32')
    rgb = np.transpose(rgb, [1, 2, 0])
    # dense = np.zeros(rgb.shape[1:]) ## c, h ,w in --> hw outs
    dense = np.zeros(rgb.shape[:2] + (2,))
    # print(dense.shape)
    # dense = np.transpose(dense, [1, 2, 0])

    # dense = np.expand_dims(dense, axis=0) ## new placeholder for color outpt

    black = np.array([1,0])
    white = np.array([0,1])

    # blackcount = 0
    # whitecount = 0

    for label, color in enumerate(colorDict.keys()):
        ## whever cell region is, make black; else white. 
        ## since cell regions are first three colors from dictionary, will use those. 
        color = np.array(color)
        # print(rgb.shape, color.shape, dense.shape)
        if label < (len(colorDict.keys())- 1):
            # print(dense.shape, rgb.shape, color.shape, label)
            dense[np.all(rgb == color, axis=-1)] = black
            # blackcount += np.sum(np.all(rgb == color, axis=-1))
        else:
            dense[np.all(rgb == color, axis=-1)] = white
            # whitecount += np.sum(np.all(rgb == color, axis=-1))
        
    
    
    # print('blackcount', blackcount, 'whitecount', whitecount)
    
    return np.transpose(dense, [2, 0, 1])

## instead of usign one-hot encoding, use sparse encoding
def rgbToOnehotSparse(rgb, colorDict):
    rgb = np.array(rgb).astype('int32')
    dense = np.zeros(rgb.shape[:2])
    for label, color in enumerate(colorDict.keys()):
        color = np.array(color)
        if label < len(colorDict.keys()):
            dense[np.all(rgb == color, axis=-1)] = label

    return dense


def outputClassImages(onehot, color_dict):
    onehot = np.array(onehot)
    onehot = np.argmax(onehot, axis=1)
    onehot = np.transpose(onehot, [1, 2, 0])

    # numClasses = len(color_dict.keys())
    outputList = []  # temporarily store each image as it is created
    totalOutput = 255*np.ones(onehot.shape[0:2]+(3,))

    for i, k in enumerate(color_dict.keys()):
        output = 255*np.zeros(onehot.shape[0:2])
        if i < len(color_dict.keys()) - 1:  # excluding background class (last entry in list)
            totalOutput[np.all(onehot == i, axis=2)] = k
            output[np.all(onehot == i, axis=-1)] = 255
            outputList.append(output)

    outputList.append(totalOutput)
    return outputList


## instead of modifying class based output, will simply convert to bw whenever a non-background class is present. Used this when transferring from MC to binary
def RGBtoBW(image, isGrayscale=False):
    image = image.astype('uint8')
    bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype('uint8')
    ret, bw = cv2.threshold(bw,254,255,cv2.THRESH_BINARY)  ## threshold to combine all colors in output channel
    
    if isGrayscale:
        bw = np.expand_dims(bw, axis=-1)  ## making 3-d tupule
        bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    
    return bw



def onehotToIMG(img, bw=False):
    img = np.array(img)
    output = np.zeros(img.shape[:2] + (3,))
    img = np.argmax(img, axis=-1, keepdims=True)

    # print('img shape', img.shape)
    # print('zeros', np.sum(np.all(img == 0, axis=-1)))
    # print('ones', np.sum(np.all(img == 1, axis=-1)))

    output[np.all(img == 0, axis=-1)] = [0, 0, 0]
    output[np.all(img == 1, axis=-1)] = [255, 255, 255]

    if bw:
        output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_RGB2GRAY)

    return output