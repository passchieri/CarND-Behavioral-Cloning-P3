#!/usr/bin/env python

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sklearn.utils



########Read images
Xbase = []
ybase = []

files=[]
flips=[]
stears=[]

def import_dir(dir):
    lines=[]
    print ('processing directory '+dir)
    logfile=dir+'/driving_log.csv'
    assert os.path.exists(logfile)
    with open(logfile) as f:
        reader=csv.reader(f)
        first=True
        for line in reader:
            if (not first):
                 lines.append(line)
            first=False

    print('found {} potential images'.format(len(lines)))
    for line in lines:
        stear=float(line[3])

        #center image
        path=line[0]
        fname=dir+'/IMG/'+path.split('/')[-1]
        if (os.path.exists(fname)):
            files.append(fname)
            flips.append(False)
            stears.append(stear)
            files.append(fname)
            flips.append(True)
            stears.append(-1.*stear)

            #left image
            path=line[1]
            fname=dir+'/IMG/'+path.split('/')[-1]
            files.append(fname)
            stears.append(stear+0.4)
            flips.append(False)
            files.append(fname)
            flips.append(True)
            stears.append(-1.*stear-0.4)

            #right image
            path=line[2]
            fname=dir+'/IMG/'+path.split('/')[-1]
            files.append(fname)
            flips.append(False)
            stears.append(stear-0.4)
            files.append(fname)
            flips.append(True)
            stears.append(-1.*stear+0.4)
        else:
            print ('skipping non existing images: '+path.split('/')[-1])


def generator(files,flips ,stears, batch_size=32):
    num_samples = len(files)
    assert (len(files)==len(stears) and len(files)==len(flips))
    while 1:
        sklearn.utils.shuffle(files,stears,flips)
        for offset in range(0, num_samples, batch_size):
            batch_files=files[offset:offset+batch_size]
            batch_flips=flips[offset:offset+batch_size]
            batch_stears=stears[offset:offset+batch_size]

            images=[]
            angles=[]
            for i in range(len(batch_files)):
                fname=batch_files[i]
                stear=batch_stears[i]
                flip=batch_flips[i]
                img=cv2.imread(fname)
                if (flip):
                    img= np.fliplr(img)
                images.append(img)
                angles.append(stear)

            X_train=np.array(images)
            y_train=np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


import_dir('data/example')
import_dir('data/run1')

from sklearn.model_selection import train_test_split
train_files, val_files,train_flips,val_flips,train_stears,val_stears = train_test_split(files,flips,stears, test_size=0.2)

train_generator = generator(train_files,train_flips,train_stears, batch_size=64)
val_generator = generator(val_files,val_flips,val_stears, batch_size=64)


###########build model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
from keras.callbacks import EarlyStopping,LambdaCallback
from keras.backend import tf as ktf

model=Sequential()
#model.add(Flatten(input_shape=X.shape[1:]))
#model.add(Lambda(lambda image: ktf.image.resize_images(image, (32, 32)),input_shape=X.shape[1:]))
model.add(Lambda(lambda x: (x/127.5)-1.0,input_shape=[160,320,3]))
##Lenet
#model.add(Cropping2D(cropping=((50,20), (0,0))))
#model.add(Convolution2D(6,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.1))
#model.add(Convolution2D(16,5,5,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.1))
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))


model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Convolution2D(36,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.1))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

##############train model
model.compile(loss='mse',optimizer='adam')
hist=model.fit_generator(train_generator,
        samples_per_epoch=len(train_files),
        validation_data=val_generator,
        nb_val_samples=len(val_files), nb_epoch=20, 
        callbacks=[EarlyStopping(monitor='val_loss', patience=1, verbose=1),
	LambdaCallback(on_epoch_end=lambda epoch,logs:model.save('temp.h5'))])

model.save('model.h5')
print ("loss:")
print (hist.history['loss'])
print ("val_loss:")
print (hist.history['val_loss'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

