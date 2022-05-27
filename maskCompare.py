import cv2
from skimage.metrics import structural_similarity
import numpy as np
import os

class ImageComparator():
    def __init__(self, pathToFiles):
        self.globalPath = pathToFiles
        for folder in self.globalPath:
            pass



# maskOriginal = cv2.imread("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/mask1.png")
# maskNet = cv2.imread("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/WMH1_epochs1000_efficientnet.png")
# brain = cv2.imread("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/brain1.png")

    def importFiles(self, pathMaskOriginal, pathMaskNet, pathBrain):
        self.maskOriginal = cv2.imread(pathMaskOriginal)
        self.maskNet = cv2.imread(pathMaskNet)
        self.brain = cv2.imread(pathBrain)

    def compareProcess(self, save):
        imgRED = self.maskNet
        imgRED[np.where((imgRED==[255,255,255]).all(axis=2))] = [0,0,255]

        imgBLUE = self.maskOriginal
        imgBLUE[np.where((imgBLUE==[255,255,255]).all(axis=2))] = [255,0,0]


        imgGreen = cv2.add(imgRED, imgBLUE)
        imgGreen[np.where((imgGreen==[255,0,255]).all(axis=2))] = [0,255,0]
        imgGreen[np.where((imgGreen==[255,0,0]).all(axis=2))] = [0, 255, 255]



        vis2 = cv2.addWeighted(imgGreen, 0.6, self.brain, 0.8, 0)

        vis2 = cv2.resize(vis2,(512, 512))


        cv2.imshow('filled after', vis2)
        cv2.waitKey()

if __name__ == "__main__":
    unit = ImageComparator("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/mask1.png", "D:/Projekty/Git projekt/mastery-machine-learning/wyniki/WMH1_epochs1000_efficientnet.png", "D:/Projekty/Git projekt/mastery-machine-learning/wyniki/brain1.png")


