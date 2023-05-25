## use this script to shuffle dataset, and if desired, output new folders. 
import os
import random
import cv2

def shuffleDataset(rootDir, outputDir, shuffle=True, trainVal = False, valSplit=0.2):
    '''
    used this to create train, validation split and corresponding folders. Should only have to be done once, after which files will be in correct location for dataloader. 
    '''
    
    ## ensure output directory exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    ## ensure output directory has images and masks
    if trainVal:
        if not os.path.exists(outputDir + 'Train/'):
            os.makedirs(outputDir + 'Train/')
        if not os.path.exists(outputDir + 'Val/'):
            os.makedirs(outputDir + 'Val/')
        if not os.path.exists(outputDir + 'Train/Masks/'):
            os.makedirs(outputDir + 'Train/Masks/')
        if not os.path.exists(outputDir + 'Train/Images/'):
            os.makedirs(outputDir + 'Train/Images/')
        if not os.path.exists(outputDir + 'Val/Masks/'):
            os.makedirs(outputDir + 'Val/Masks/')
        if not os.path.exists(outputDir + 'Val/Images/'):
            os.makedirs(outputDir + 'Val/Images/')
    else:
        if not os.path.exists(outputDir + 'Masks/'):
            os.makedirs(outputDir + 'Masks/')
        if not os.path.exists(outputDir + 'Images/'):
            os.makedirs(outputDir + 'Images/')
        
    
    imlist = os.listdir(rootDir + '/Images')

    numImgs = len(imlist)

    ## shuffle dataset
    for i in range(42):
        random.shuffle(imlist)

    trainlist = []
    vallist = []
    
    trainimages = []
    trainmasks = []
    valimages = []
    valmasks = []

    ## ensure both train and mask images are loaded correctly 

    ## if want to create train/ validation split, use this loop
    if trainVal:
        for iter, im in enumerate(imlist):
            if iter < numImgs * (1 - valSplit):
                trainlist.append(im)
            else:
                vallist.append(im)
            
        for im in trainlist:
            img = cv2.imread(rootDir + 'Images/' + im)
            cv2.imwrite(outputDir + 'Train/Images/' + im, img)
            img = cv2.imread(rootDir + 'Masks/' + im)
            cv2.imwrite(outputDir + 'Train/Masks/' + im, img)
        for im in vallist:
            img = cv2.imread(rootDir + 'Images/' + im)
            cv2.imwrite(outputDir + 'Val/Images/' + im, img)
            img = cv2.imread(rootDir + 'Masks/' + im)
            cv2.imwrite(outputDir + 'Val/Masks/' + im, img)

        
    else:
        for im in imlist:
            img = cv2.imread(rootDir + 'Images/' + im)
            cv2.imwrite(outputDir + 'Images/' + im, img)
            img = cv2.imread(rootDir + 'Masks/' + im)
            cv2.imwrite(outputDir + 'Masks/' + im, img)


def main():
    rootPath = '/home/noahvandal/my_project_dir/my_project_env/TrainUNET/Dataset/'

    sourcePath = rootPath + 'FineDataset/'
    destPath = rootPath + 'FineSplit/'

    shuffleDataset(sourcePath, destPath, trainVal=True, valSplit=0.2)


if __name__ == '__main__':
    main()
        
            
        
