import os
import cv2
import numpy as np

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            preprocessors = []

    def load(self, imagePaths):
        data = []
        labels = []

        for (i,imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            #label = imagePath.split(os.path.sep)[-2]
            imageFileName = imagePath.split(os.path.sep)[-1]
            label = imageFileName.split('.')[0]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)

        return (np.array(data), np.array(labels))