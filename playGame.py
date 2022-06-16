import os
import pandas as pd
from mkdir_p import mkdir_p
import time
from PIL import Image
import pyautogui
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from train import final_model
from functions import avanzarframe, prepare_image, screenshot, gameinfo

comands = ['R','L','U','D','J','X','Z','G','S']

def introTas(name):
    pyautogui.write('console hard 1 0.0 144.0')
    pyautogui.press('enter')
    pyautogui.write(f'StartExportGameInfo run_{name}.txt')
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
    alttab()
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
    mkdir_p("weights")
    weights_file = "weights/modelo1.hdf5"
    model = final_model()
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)
    mkdir_p("gamesPlayed")
    game_file = f"gamesPlayed/{name}.tas"
    if os.path.isfile(game_file):
        answer = pyautogui.confirm(text='The name of the new game already exists in the directory. Do you want to overwrite it?',
         buttons=['OK', 'Cancel'], title='ATTENTION')
        if(answer=='Cancel'): 
            return "The run doesn't start because the run name already exists"
    pyautogui.alert('Remenber to set Celeste Studio in the front with a new clear document. It is necesary too to have Celeste to be the previous window so we can tab it. Press OK to start.')
    time.sleep(3)
    introTas(name)
    pyautogui.hotkey('alt','tab', interval=0.1)
    time.sleep(0.1)
    pyautogui.click()
    pyautogui.keyDown('o')
    pyautogui.keyUp('o')
    pyautogui.keyDown('p')
    pyautogui.keyUp('p')
    avanzarframe(36)
    frames=39
    while frames < 60:
            image = prepare_image(screenshot())
            info = pd.read_csv("CelesTAS/dump.txt", delimiter='\t',index_col='Frames')
            info = info.drop(columns=['Line','Entities'])
            posX, posY, spdX, spdY, state, statuses, inputs = gameinfo(frames, info)
            if(posX=='skip'):
                avanzarframe(2)
                frames = frames + 2
                continue
            else:
                imageaux = np.asarray(image)
                aux = [float(posX), float(posY), float(spdX), float(spdY), state]
                aux= np.asarray([aux +statuses])
                sol = model.predict(x=[aux,imageaux])
                sol = sol[0].tolist()
                for i,  value in enumerate(sol):
                    sol[i]=round(value)
                writeTas(sol)
                if 'Dead' in statuses:
                    death = True
                if (float(posX) == 4991.1264) & (float(posY) == -3202.0181):
                    end_level = True
                avanzarframe(2)
                frames = frames + 2

if __name__ == '__main__':
    playTime('x', 'y')