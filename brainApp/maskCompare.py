import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

class ImageComparator():
    def __init__(self, pathToFiles):
        self.globalPath = pathToFiles
        self.listOfDirectories = os.listdir(pathToFiles)
        self.minimum_commutative_image_diff = 1

# maskOriginal = cv2.imread("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/mask1.png")
# maskNet = cv2.imread("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/WMH1_epochs1000_efficientnet.png")
# brain = cv2.imread("D:/Projekty/Git projekt/mastery-machine-learning/wyniki/brain1.png")

    def importFiles(self, directory):
        pathMaskOriginal = directory + "/mask.png"
        pathMaskNet = directory + "/seg.png"
        pathBrain = directory + "/brain.png"

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
            cv2.imwrite("processingImg.png", vis2)

        if show:
            cv2.imshow('DATA', vis2)
            cv2.waitKey()

    def dataProcessing(self):
        mask = cv2.cvtColor(self.maskOriginal, cv2.COLOR_BGR2GRAY)
        net = cv2.cvtColor(self.maskNet, cv2.COLOR_BGR2GRAY)
        mask = np.asarray(mask, dtype="int32")
        net = np.asarray(net, dtype="int32")

        counterDiv = 0
        div = np.subtract(mask, net)
        divNumber = np.where(div == -255)
        x, divNumber = divNumber.shape

    def compareImage(self):
        

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff






    def compareProcess(self, save):
        for directory in self.listOfDirectories:
            self.importFiles(directory=directory)
            self.imageProcessing()






if __name__ == "__main__":

    unit = ImageComparator("/home/arkadiusz/Documents")


