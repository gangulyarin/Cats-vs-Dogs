import argparse
from imutils import paths
from simpledatasetloader import SimpleDatasetLoader
from Preprocessing.simplepreprocessor import SimplePreprocessor
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True, help='path to input datsset')
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32,32)
sd = SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sd.load(imagePaths)
data = data.reshape((data.shape[0],(32*32*3)))

le = LabelEncoder()
labels_transformed = le.fit_transform(labels)

knn = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])

train_X, test_X, train_y, test_y = train_test_split(data,labels_transformed)

knn.fit(train_X,train_y)

print(classification_report(test_y,knn.predict(test_X),target_names=le.classes_))

