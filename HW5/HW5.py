# -*- coding: utf-8 -*-
"""HW5.ipynb"""

import tensorflow as tf
import numpy as np                                
import matplotlib.pyplot as plt
import keras
from keras import models
from keras import layers
from keras import regularizers
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam # - Works
from keras.regularizers import l2
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
import h5py
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


#path = '/mnt/HDD2/Tzuchi/Pattern_recognition/HW5/'
path = ''
#model_path = 'cifar_model_resnet50'
model_path = 'cifar_model_VGG'
#model_path = 'VGG_model_cifar_20epoch'

def DNN_model(reg=l2(1e-4), img_rows=32, img_cols=32, 
                channels=3, num_filters=32, ac='relu', drop_dense=0.5, drop_conv=0, num_classes=10):
    # build and compile the model  (roughly following the VGG paper)

    #reg=l2(1e-4)   # L2 or "ridge" regularisation
    
    model = Sequential()

    model.add(Conv2D(num_filters, (3, 3), activation=ac, kernel_regularizer=reg, input_shape=(img_rows, img_cols, channels),padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 16x16x3xnum_filters
    model.add(Dropout(drop_conv))

    model.add(Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 8x8x3x(2*num_filters)
    model.add(Dropout(drop_conv))

    model.add(Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 4x4x3x(4*num_filters)
    model.add(Dropout(drop_conv))

    model.add(Flatten())
    model.add(Dense(512, activation=ac,kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Dropout(drop_dense))
    model.add(Dense(num_classes, activation='softmax'))

    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=keras.optimizers.adamax(lr=5e-3))

    model.summary()
    return model

def ResNet_model():
    model = ResNet50(weights=None, classes=10, input_shape=(32,32,3))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# plotting helper function
def plothist(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('History.png')
    plt.show()


def Save_and_load_model(model):
    # Saving and loading model and weights
    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example

    model_json = model.to_json()


    with open(path + model_path + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(path + model_path + ".h5")

    # load json and create model
    json_file = open(path +  model_path + ".json", 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(path +  model_path + ".h5")
    print("Loaded model from disk")

    loaded_model.save(path +  model_path + '.hdf5')
    loaded_model=load_model(path +  model_path + '.hdf5')

    return loaded_model

def Load_model(model):
    loaded_model=load_model(path +  model_path + '.hdf5')
    return loaded_model


if __name__ == '__main__':
    x_train = np.load(path + "x_train.npy")
    y_train = np.load(path + "y_train.npy")

    x_test = np.load(path + "x_test.npy")
    y_test = np.load(path + "y_test.npy")

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # It's a multi-class classification problem 
    class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                'dog': 5, 'frog': 6,'horse': 7,'ship': 8, 'truck': 9}
    print(np.unique(y_train))
    '''
    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i])
    # show the figure
    plt.savefig('Image/Example_9_image.png')
    plt.show()

    im = Image.fromarray(x_train[4])
    #im.save(path + "test_image/Example_car.png")

    #Loads image in from the set image path
    img_path = path + "test_image/Example_car.png"
    img = keras.preprocessing.image.load_img(img_path, target_size= (32,32))
    img_tensor = keras.preprocessing.image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    #Allows us to properly visualize our image by rescaling values in array
    img_tensor /= 255.
    #Plots image
    #plt.figure(figsize=(32,32))
    plt.imshow(img_tensor[0])
    plt.show()
    '''
    # set up image augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        #vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        #zoom_range=0.3
        )
    '''
    #Creates our batch of one image
    pic = datagen.flow(img_tensor, batch_size =1)
    #plt.figure(figsize=(32*3,32*3))
    #Plots our figures
    for i in range(1,4):
        plt.subplot(1, 3, i)
        batch = pic.next()
        image_ = batch[0].astype('uint8')
        plt.imshow(image_)
    plt.savefig('Image/Example_data_augentation_image.png')
    plt.show()
    '''
    ## Data preprocess

    # Normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to one-hot encoding (keras model requires one-hot label as inputs)
    num_classes = 10
    print(y_train[0])
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    print(y_train[0])

    model = DNN_model(num_classes=num_classes)
    #model = ResNet_model()
    
    if os.path.exists(model_path + '.hdf5'):
        loaded_model = Load_model(model)
    else:
        # Split the data
        x_train_split, x_valid, y_train_split, y_valid = train_test_split(x_train, y_train, test_size=0.15, shuffle= True)
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        # train with image augmentation
        history = model.fit_generator(datagen.flow(x_train_split, y_train_split, batch_size=128),
                            steps_per_epoch = len(x_train_split) / 128, epochs=100, validation_data=(x_valid, y_valid))
        plothist(history)  # 128 batch, 0.001 lr, 512 neurons, no zoom, no convdrop, only 0.1 shift
        loaded_model = Save_and_load_model(model)
    
    
    y_pred = loaded_model.predict(x_test)
    y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(float)

    print(y_pred.shape) # 10000 samples, each sample with probaility of 10 classes
    assert y_pred.shape == (10000, 10)
    print("Accuracy of my model on test set: ", accuracy_score(y_test, y_pred))

    """## DO NOT MODIFY CODE BELOW!
    **Please screen shot your results and post it on your report**
    """

    '''
    y_pred = model.predict(x_test)

    assert y_pred.shape == (10000,)

    y_test = np.load("y_test.npy")
    print("Accuracy of my model on test set: ", accuracy_score(y_test, y_pred)

    '''



