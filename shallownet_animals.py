import argparse
from imutils import paths
from simpledatasetloader import SimpleDatasetLoader
from Preprocessing.simplepreprocessor import SimplePreprocessor
from Preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from conv.shallownet import ShallowNet
from sklearn.preprocessing import LabelBinarizer

ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True, help='path to input datsset')
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()
sd = SimpleDatasetLoader(preprocessors=[sp,iap])
(data,labels) = sd.load(imagePaths)
#data = data.reshape((data.shape[0],(32*32*3)))
data = data.astype("float")/255.0

#le = LabelEncoder()
#labels_transformed = le.fit_transform(labels)

lb = LabelBinarizer()
labels_transformed = lb.fit_transform(labels)

print(labels_transformed.shape)

train_X, test_X, train_y, test_y = train_test_split(data,labels_transformed)

model = ShallowNet.build(32,32,3,classes=1)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(train_X,train_y,batch_size=32,epochs=3)

predictions = model.predict(test_X,32)

print(classification_report(test_y.argmax(axis=1),predictions.argmax(axis=1),target_names=lb.classes_))