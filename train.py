import argparse
import os
import sys
import pandas as pd
import numpy as np
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          concatenate)
from keras.models import Model, Sequential
from mkdir_p import mkdir_p
from tensorflow import keras

from functions import (grabar_juego, parseConfig)

# Number of output neurons. 
#In order, R - L - U - D - J - X - Z - G - S
OUT_SHAPE = 9

#Aspect ratio 16:9. We are looking for nHD standard resolution
INPUT_WIDTH = 640
INPUT_HEIGHT = 360
INPUT_CHANNELS = 3

# Percentage of validation split. Set to a higher value when you have large training data.
VALIDATION_SPLIT = 0.15
# Data augmentation not tested
USE_REVERSE_IMAGES = False

# Create MLP - Dim: 13
def create_mlp():
    model = Sequential()
    model.add(Dense(8, input_dim=13, activation="relu"))
    model.add(Dense(4, activation="relu"))
    return model

# Create CNN, check kep_prob parameter since it controls dropout layers
def create_cnn(keep_prob=0.6):

    # Keras sequential model
    model = Sequential()

    # Input layer with defined size
    model.add(BatchNormalization(input_shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)))

    # Convolutional layers
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(12896, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(6448, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(3223, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(1074, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(358, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(drop_out))

    #CNN Output layer
    model.add(Dense(32, activation='relu', name="predictions"))
    return model

def final_model():
    mlp = create_mlp()
    cnn = create_cnn()

    combinedInput = concatenate([mlp.output, cnn.output])
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(18, activation="relu")(combinedInput)
    x = Dense(OUT_SHAPE, activation="sigmoid")(x)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)

    return model


# Load images and steering files from recordings folder

def load_training_data():
    data = grabar_juego()
    x_train_images = []
    x_train_info = []
    y_train = []
    for row in data: 
        y_train.append(row[7])
        row.pop()
        x = row[0]
        x_train_images.append(x[0])
        aux = [float(row[1]),float(row[2]),float(row[3]),float(row[4]),row[5]]
        list_aux = row[6]
        x_train_info.append(aux + list_aux)
    x_train_images = np.array(x_train_images)
    x_train_info = np.array(x_train_info)
    y_train = np.array(y_train)
    return x_train_images,x_train_info, y_train, x_train_images,x_train_info, y_train



if __name__ == '__main__':
    config = pd.read_csv("config.txt", delimiter='=',index_col='variable')

    parser = argparse.ArgumentParser()
    parser.add_argument('model', default="model_default")
    parser.add_argument('-c', '--cpu', action='store_true', help='Force Tensorflow to use the CPU.', default=False)
    args = parser.parse_args()
    # Load Training Data
    X_train_images,X_train_info, y_train,X_val_images, X_val_info, y_val = load_training_data()

    # Training loop variables
    try:
        epochs = int(parseConfig("epochs", 20))
        batch_size = int(parseConfig("batchSize", 20))
    except:
        sys.exit("The config value of the frames or the end points are not integers. Check it and try again.")
    
    loss = parseConfig("lossFunction", keras.losses.cosine_similarity)
    optimizer = parseConfig("optimizerFunction", 'adam')

    model = final_model()

    mkdir_p("weights")
    weights_file = "weights/{}.hdf5".format(args.model)
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    model.compile(loss=loss, optimizer= optimizer)
    model.fit(x=[X_train_info, X_train_images],y=y_train, batch_size=batch_size, epochs=epochs,
            verbose=True)
    model.save("weights/{}.hdf5".format(args.model))
    print("Model saved as {}.hdf5".format(args.model))


