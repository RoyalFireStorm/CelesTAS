import glob
import os
import hashlib
import time
import argparse
from mkdir_p import mkdir_p

from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Concatenate, Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
from functions import grabar_juego
# Number of output neurons. 
#In order, R - L - U - D - J - X - Z - G - S
OUT_SHAPE = 9

# Height and Width of model imput. For RGB inmages, use 3 channels
INPUT_WIDTH = 955
INPUT_HEIGHT = 530
INPUT_CHANNELS = 3

# Percentage of validation split. Set to a higher value when you have large training data.
VALIDATION_SPLIT = 0.15
# Data augmentation not tested
USE_REVERSE_IMAGES = False

def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
              + K.sum(K.square(y_pred), axis=-1) / 2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = K.sqrt(K.sum(K.square(float(y_pred) - float(y_true)), axis=-1))
    return val

# Create CNN, check kep_prob parameter since it controls dropout layers
def create_model(keep_prob=0.6):

    #Aux
   # aux_input = Input(shape=(3,))
    # Keras sequential model
    model = Sequential()

    # Input layer with defined size
    model.add(BatchNormalization(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))

    # Convolutional layers
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # Dense layers
    model.add(Flatten())
   # model.add(Concatenate(aux_input))
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))

    # Output layer
    model.add(Dense(OUT_SHAPE, activation='softsign', name="predictions"))

    return model

# Load images and steering files from recordings folder

def load_training_data():
    data = grabar_juego()
    x_train = []
    y_train = []
    for row in data: 
        y_train.append(row[7])
        row.pop()
        x = row[0]
        x_train.append(x[0])
    #Aqui pondré los i_val necesarios posteriormente
    return x_train, y_train, x_train, y_train


if __name__ == '__main__':

    '''parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()'''
    # Load Training Data
    X_train, y_train, X_val, y_val = load_training_data()

    '''print(X_train.shape[0], 'training samples.')
    print(X_val.shape[0], 'validation samples.')'''

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model()

    mkdir_p("weights")
    weights_file = "weights/modelo1.hdf5"'''.format(args.model)'''
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    model.compile(loss=customized_loss, optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001))
    checkpointer = ModelCheckpoint(
        monitor='val_loss', filepath=weights_file, verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True, callbacks=[checkpointer, earlystopping])
    model.save("weights/modelo1.hdf5"'''.format(args.model)''')
    #validation_data=(X_val, y_val),
    