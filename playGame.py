import os
import sys
import pandas as pd
from mkdir_p import mkdir_p
import time
import psutil
from PIL import Image
import pyautogui
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from train import final_model
from functions import avanzarframe, capture_info, prepare_image, screenshot, gameinfo, select_celeste_path

comands = ['R','L','U','D','J','X','Z','G','S']

def introTas(name, model_name):
    pyautogui.write('console hard 1 0.0 144.0')
    pyautogui.press('enter')
    pyautogui.write('1')
    pyautogui.press('enter')
    pyautogui.write(f'StartExportGameInfo run_{name}_{model_name}.txt')
    pyautogui.press('enter')
    pyautogui.write('2')
    pyautogui.press('enter')
    pyautogui.write('40')
    pyautogui.press('enter')
    

def alttab():
    pyautogui.hotkey('alt','tab', interval=0.1)
    time.sleep(0.1)
    

def writeTas(prediction):
    #In order, R - L - U - D - J - X - Z - G - S
    
    pyautogui.press('enter')
    row=[]
    for i, bool in enumerate(prediction):
        if(bool==1):
            row.append(comands[i])
    pyautogui.write('2')
    for entry in row:
        pyautogui.write(entry)
        time.sleep(0.05)
    
    time.sleep(0.05)
    alttab()

def playTime(model_name, name):
    #file_dir = select_celeste_path()[:-11] + f"run_{name}_{model_name}.txt"
    mkdir_p("weights")
    weights_file = f"weights/{model_name}.hdf5"
    model = final_model()
    if os.path.isfile(weights_file) == False:
        sys.exit("The model called " + model_name + " does not exists.")    
    model.load_weights(weights_file)
    mkdir_p("gamesPlayed")
    game_file = f"gamesPlayed/{name}_{model_name}.tas"
    if os.path.isfile(game_file):
        answer = pyautogui.confirm(text='The name of the new game already exists in the directory. Do you want to overwrite it?',
         buttons=['OK', 'Cancel'], title='ATTENTION')
        if(answer=='Cancel'): 
            return "The run doesn't start because the run name already exists"
    pyautogui.alert('Remenber to set Celeste Studio in the front with a new clear document. It is necesary too to have Celeste to be the previous window so we can tab it. Press OK to start.')
    time.sleep(3)
    introTas(name, model_name)
    pyautogui.hotkey('alt','tab', interval=0.1)
    time.sleep(0.1)
    pyautogui.click()
    pyautogui.keyDown('o')
    pyautogui.keyUp('o')
    pyautogui.keyDown('p')
    pyautogui.keyUp('p')
    avanzarframe(1)
    alttab()
    time.sleep(1)
    im = pyautogui.screenshot(region=(0,900, 350, 135))
    frames= capture_info(im)[0]
    avanzarframe(42 - frames)
    alttab()
    time.sleep(0.3)
    while frames < 100:
            image = prepare_image(screenshot())
            alttab()
            im = pyautogui.screenshot(region=(0,900, 350, 135))
            frames, posX, posY, spdX, spdY, state, statuses = capture_info(im)
            imageaux = np.asarray(image)
            aux = [posX, posY, spdX, spdY, state]
            aux= np.asarray([aux +statuses])
            sol = model.predict(x=[aux,imageaux])
            sol = sol[0].tolist()
            for i,  value in enumerate(sol):
                sol[i]=round(value)
            writeTas(sol)
            if 'Dead' in statuses:
                death = True
            if (5040 <= int(float(posX)) <= 5056) & (-3266 >= int(float(posY)) + 12 >= -3280):
                end_level = True
            avanzarframe(2)
            frames = frames + 2
            print(frames)

if __name__ == '__main__':
    playTime('modelo5', 'a')