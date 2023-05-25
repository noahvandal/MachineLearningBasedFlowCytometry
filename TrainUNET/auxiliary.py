import cv2
import numpy as np
import torch


def accuracy(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    # print(output.shape, target.shape)
    output = np.argmax(output, axis=1)
    output = np.where(output > 0.5, 1, 0)
    target = np.argmax(target, axis=1)
    # print(output.shape, target.shape)
    # print(output, target)

    tp, tn, fp, fn = truthValues(output, target)
    # print(tp, tn, fp, fn)
    precision, recall, f1, acc = precisionRecallF1Acc(tp, tn, fp, fn)
    # print(precision, recall, f1, acc)

    xorImage = np.logical_xor(output, target)

    return precision, recall, f1, acc, xorImage, [tp, tn, fp, fn]


def truthValues(matA, matB):
    tp =  np.abs(np.sum(np.logical_and(matA == 1, matB == 1)))
    tn =  np.abs(np.sum(np.logical_and(matA == 0, matB == 0)))
    fp =  np.abs(np.sum(np.logical_and(matA == 1, matB == 0)))
    fn =  np.abs(np.sum(np.logical_and(matA == 0, matB == 1)))

    return tp, tn, fp, fn


def precisionRecallF1Acc(tp, tn, fp, fn):
    epsilon = 1 ## prevent from dividing by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall) # no epsilon here
    acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return precision, recall, f1, acc


def IoU(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    output = np.argmax(output, axis=1)
    # output = np.where(output > 0.5, 1, 0)
    target = np.argmax(target, axis=1)
    # print(output.shape, target.shape)

    #cell iou
    intersectCell = np.sum(np.logical_and(output == 0, target == 0))
    unionCell = np.sum(np.logical_or(output == 0, target == 0))
    iouCell = intersectCell / unionCell
    # print(intersectCell, unionCell)

    ## background iou
    intersectBack = np.sum(np.logical_and(output == 1, target == 1))
    unionBack = np.sum(np.logical_or(output == 1, target == 1))
    iouBack = intersectBack / unionBack
    # print(intersectBack, unionBack)

    # print(intersectCell, unionCell, intersectBack, unionBack)
    # print(iouCell, iouBack)

    return [iouCell, intersectCell, unionCell, iouBack]


def calculateActualValue(value, batchsize, stepsize):
    return (value / stepsize) / batchsize