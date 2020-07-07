import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D
from keras.models  import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt

import shutil
import os
if os.path.exists("spotgarbage-GINI/spotgarbage/ambiguous-annotated-images/"):
    shutil.rmtree('spotgarbage-GINI/spotgarbage/ambiguous-annotated-images/')

dir_path1= '/spotgarbage-GINI/spotgarbage/garbage-queried-images'
dir_path2= '/spotgarbage-GINI/spotgarbage/non-garbage-queried-images'

dir_path= 'spotgarbage-GINI/spotgarbage/'
img_list = glob.glob(os.path.join(dir_path,'**/*.jpg'),recursive=True)
len(img_list)

train=ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         validation_split=0.1,
                         rescale=1./255,
                         shear_range = 0.1,
                         zoom_range = 0.1,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,)

test=ImageDataGenerator(rescale=1/255,
                        validation_split=0.1)

train_generator=train.flow_from_directory(dir_path,
                                          target_size=(300,300),
                                          batch_size=32,
                                          class_mode='categorical', #class_mode='binary'
                                          shuffle=True,
                                          seed=42,
                                          subset='training')

test_generator=test.flow_from_directory(dir_path,
                                        target_size=(300,300),
                                        batch_size=32,
                                        class_mode='categorical', #class_mode='binary'
                                        shuffle=True,
                                        seed=42,
                                        subset='validation')

labels = (train_generator.class_indices)
print(labels)

labels = dict((v,k) for k,v in labels.items())
print(labels)

for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape

print (train_generator.class_indices)

Labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('labels.txt', 'w') as f:
  f.write(Labels)

model=Sequential()
#Convolution blocks

model.add(Conv2D(32,(3,3), padding='same',input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
#model.add(SpatialDropout2D(0.5)) # No accuracy

model.add(Conv2D(64,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 
#model.add(SpatialDropout2D(0.5))

model.add(Conv2D(32,(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2)) 

#Classification layers
model.add(Flatten())

model.add(Dense(64,activation='relu'))
#model.add(SpatialDropout2D(0.5))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))

filepath="trained_model.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']) # RMS PROP - No accuracy

history = model.fit_generator(train_generator,
                              epochs=10,
                              steps_per_epoch=2276//32,
                              validation_data=test_generator,
                              validation_steps=251//32,
                              workers = 4,
                              callbacks=callbacks_list)

from keras.preprocessing import image

def classify_image(img):
    img_path = img

    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img=np.array(img)/255.0

    plt.title("Loaded Image")
    plt.axis('off')
    plt.imshow(img.squeeze())

    p=model.predict(img[np.newaxis, ...])

    #print("Predicted shape",p.shape)
    print("Maximum Probability: ",np.max(p[0], axis=-1))
    predicted_class = labels[np.argmax(p[0], axis=-1)]
    print("Classified:",predicted_class)

classify_image('spotgarbage-GINI/spotgarbage/garbage-queried-images/city garbage/398faec8-6799-11e5-8dc4-40f2e96c8ad8.jpg')

classify_image('spotgarbage-GINI/spotgarbage/non-garbage-queried-images/Indian+roads/00ca7a52-943d-11e5-9331-40f2e96c8ad8.jpg')

def preprocess(img):
    img_path = img

    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img=np.array(img)/255.0
    return img[np.newaxis, ...]

from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(model, 'my_model.pkl') 
  
# Load the model from the file 
garbage_from_joblib = joblib.load('my_model.pkl')  
  
# Use the loaded model to make predictions 
# garbage_from_joblib.predict(test_generator) 

p = garbage_from_joblib.predict(preprocess("spotgarbage-GINI/spotgarbage/non-garbage-queried-images/Indian+roads/00ca7a52-943d-11e5-9331-40f2e96c8ad8.jpg")) 

print("Maximum Probability: ",np.max(p[0], axis=-1))
predicted_class = labels[np.argmax(p[0], axis=-1)]
print("Classified:",predicted_class)

# model.save('my_model.h5')

# Using saved model
from PIL import Image
from sklearn.externals import joblib 
import numpy as np
from keras.preprocessing import image
from urllib.request import urlopen

def preprocess(img):
    img_path = img
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img, dtype=np.uint8)
    img=np.array(img)/255.0
    return img[np.newaxis, ...]

model = joblib.load('my_model.pkl')

labels = {0: 'garbage-queried-images', 1: 'non-garbage-queried-images'}

url = "https://github.com/spotgarbage/spotgarbage-GINI/blob/master/spotgarbage/garbage-queried-images/garbage/2b2d41a2-6798-11e5-8c9e-40f2e96c8ad8.jpg?raw=true"
img =Image.open(urlopen(url))
img.save("out.jpg")

p = model.predict(preprocess("out.jpg")) 
print(p)
print("Maximum Probability: ",np.max(p[0], axis=-1))
predicted_class = labels[np.argmax(p[0], axis=-1)]
print("Classified:",predicted_class)