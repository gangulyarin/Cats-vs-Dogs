import argparse
from imutils import paths
from filedatasetloader import FileDatasetLoader
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from fcheadnet import FCHeadNet
from sklearn.metrics import classification_report

ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True, help='path to input datsset')
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args['dataset']))

loader = FileDatasetLoader()

df = loader.load(imagePaths)

ohe= OneHotEncoder()
le = LabelEncoder()
#df['Label'] = le.fit_transform(df['Label'])

train,test = train_test_split(df)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,rescale=1./255,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

traingen = datagen.flow_from_dataframe(train,args['dataset'],x_col='Filename',y_col='Label',target_size=(224,224),batch_size=32)

valgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_dataframe(test,args['dataset'],x_col='Filename',y_col='Label',target_size=(224,224),batch_size=32)

baseModel = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=(224,224,3)))

headModel = FCHeadNet.build(baseModel,2,256)

model = tf.keras.models.Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss="categorical_crossentropy", optimizer="rmsprop",metrics=["accuracy"])

print("Training Head...")
H = model.fit_generator(traingen,epochs=5,validation_data=valgen,validation_steps=test.shape[0]//32,steps_per_epoch=train.shape[0]//32)

print("Evaluating after initialization...")
y_pred = model.predict_generator(valgen)
print(classification_report(ohe.fit_transform(test['Label']).toarray().argmax(axis=1),y_pred.argmax(axis=1)))

for layer in baseModel.layers[:15]:
    layer.trainable = True

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=["accuracy"])

print("Fine Tuning...")
H = model.fit_generator(traingen,epochs=5,validation_data=valgen,validation_steps=test.shape[0]//32,steps_per_epoch=train.shape[0]//32)

print("Evaluating after fine tuning...")
y_pred = model.predict_generator(valgen)
print(classification_report(ohe.fit_transform(test['Label']).toarray().argmax(axis=1),y_pred.argmax(axis=1)))
