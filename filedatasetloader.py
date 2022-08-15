import os
import pandas as pd

class FileDatasetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            preprocessors = []
    
    def load(self,imagePaths):
        data = []
        for (i,imagePath) in enumerate(imagePaths):
            imageFileName = imagePath.split(os.path.sep)[-1]
            label = imageFileName.split('.')[0]
            data.append([imageFileName,label])
        
        df = pd.DataFrame(data,columns=['Filename','Label'])
        return df