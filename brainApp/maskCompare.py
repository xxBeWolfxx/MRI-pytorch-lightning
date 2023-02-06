import cv2
import numpy as np
import os
from bar import Bar

class ImageComparator():

    def __init__(self, pathToFiles):
        print("********maskComparer*************\n Trial edition")
        self.globalPath = pathToFiles + "/"
        self.listOfDirectories = os.listdir(pathToFiles)
        self.minimum_commutative_image_diff = 1


    def importFiles(self, directory):
        pathMaskOriginal = self.globalPath + directory + "/mask.png"
        pathMaskNet = self.globalPath + directory + "/seg.png"
        pathBrain = self.globalPath + directory + "/brain.png"
        self.directory = directory

        self.maskOriginal = cv2.imread(pathMaskOriginal)
        self.maskNet = cv2.imread(pathMaskNet)
        self.brain = cv2.imread(pathBrain)

    def imageProcessing(self, save = True, show = False):
        imgRED = self.maskNet
        imgRED[np.where((imgRED == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

        imgBLUE = self.maskOriginal
        imgBLUE[np.where((imgBLUE == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

        imgGreen = cv2.add(imgRED, imgBLUE)
        imgGreen[np.where((imgGreen == [255, 0, 255]).all(axis=2))] = [0, 255, 0]
        imgGreen[np.where((imgGreen == [255, 0, 0]).all(axis=2))] = [0, 255, 255]

        vis2 = cv2.addWeighted(imgGreen, 0.6, self.brain, 0.8, 0)

        vis2 = cv2.resize(vis2, (512, 512))

        if save:
            pathWrite = self.globalPath + self.directory + "/"
            cv2.imwrite(pathWrite + "processingImg.png", vis2)

        if show:
            cv2.imshow('DATA', vis2)
            cv2.waitKey()



    def compareProcess(self, save = False, show = True):
        bareczek = Bar(len(self.listOfDirectories))
        i = 0
        for directory in self.listOfDirectories:
            i = i + 1
            bareczek.process(i)
            self.importFiles(directory=directory)
            self.imageProcessing(save=save, show=show)







if __name__ == "__main__":

    unit = ImageComparator("D:/Projekty/Git projekt/mastery-machine-learning/brainApp/data")
    unit.compareProcess(save=True)


