from msilib.schema import Error
from attr import Attribute
import math
import py
import re
import pyautogui
import time
import numpy as np
from numpy import array, dtype
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sqlalchemy import Integer
import threading as th

keep_going = True
#Aspect ratio 16:9. We are looking for nHD standard resolution
INPUT_WIDTH = 640
INPUT_HEIGHT = 360
INPUT_CHANNELS = 3


def parsedInputs(inputs):
    sol = [0] * 9
    if "," in inputs:
        partes = inputs.split(',')
        partes.remove(0)
        #In order, R - L - U - D - J - X - Z - G - S
        for letter in partes:
            if(letter == 'R'):
                sol[0] = 1
            if(letter == 'L'):
                sol[1] = 1
            if(letter == 'U'):
                sol[2] = 1
            if(letter == 'D'):
                sol[3] = 1
            if(letter == 'J'):
                sol[4] = 1
            if(letter == 'X'):
                sol[5] = 1
            if(letter == 'Z'):
                sol[6] = 1
            if(letter == 'G'):
                sol[7] = 1
            if(letter == 'S'):
                sol[8] = 1
    return sol


def screenshot():
    time.sleep(0.1)
    #Take a screenshot to the region that we choose to be the game
    return pyautogui.screenshot()

def gameinfo(Frame, info):

    #Choose the frame that we want and parse the information to variables
    try:
        rowFrame = info.loc[Frame]
    except KeyError:
        return 'skip',0,0,0,0,0,0
    
    if (type(rowFrame)!=pd.Series):
        rowFrame = rowFrame.iloc[0]

    try:
        inputs = parsedInputs(rowFrame['Inputs'].strip())
    except:
        inputs = [0] * 9
        
    
    
    try:
        
        posX = rowFrame['Position'].split(',')[0].strip()
        posY = rowFrame['Position'].split(',')[1].strip()
        spdX = rowFrame['Speed'].split(',')[0].strip()
        spdY = rowFrame['Speed'].split(',')[1].strip()
    except:
        posX = 0.00
        posY = 0.00
        spdX = 0.00
        spdY = 0.00

    state = rowFrame['State']
    if(type(state) != str):
        if(np.isnan(state)):
            state = ''
    
    try:
        statuses = rowFrame['Statuses'].strip().split(' ')
    except:
        statuses = []

    """ if Frame-1 in info:
        rowPreviousFrame = info.loc[Frame-1]
        if (type(rowPreviousFrame)!=pd.Series):
            rowPreviousFrame = rowPreviousFrame.iloc[0]
        Previousstatuses = rowPreviousFrame['Statuses'].strip().split(' ')
        if 'Dead' in Previousstatuses: statuses.append('Dead') """
    

    return posX, posY, spdX, spdY, state, statuses, inputs
   
def key_capture_thread():
    global keep_going
    input()
    keep_going = False

def grabar_juego():
    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    pyautogui.FAILSAFE = True
    pyautogui.alert('Remenber to set Celeste in the front and full window screen. Press OK to start.')
    frames = 2
    data = []
    time.sleep(5)
    pyautogui.keyDown('o')
    pyautogui.keyUp('o')
    pyautogui.keyDown('p')
    pyautogui.keyUp('p')
    time.sleep(0.2)
    avanzarframe(1)
    death = False
    end_level = False
    #Read the txt and only take the important columns that we want for later
    info = pd.read_csv("dump.txt", delimiter='\t',index_col='Frames')
    info = info.drop(columns=['Line','Entities'])
    #while death == False & end_level == False:
    while frames < 100:
            image = prepare_image(screenshot())
            posX, posY, spdX, spdY, state, statuses, inputs = gameinfo(frames, info)
            if(posX=='skip'):
                avanzarframe(2)
                frames = frames + 2
                continue
            else:
                imageaux = np.asarray(image).tolist()
                aux = [imageaux, posX, posY, spdX, spdY, state, statuses, inputs]
                data.append(aux)
                if 'Dead' in statuses:
                    death = True
                if (float(posX) == 4991.1264) & (float(posY) == -3202.0181):
                    end_level = True
                avanzarframe(2)
                frames = frames + 2

    print('The recording has stopped')
    return data
def avanzarframe(num):
    i=0
    while i<num:
        pyautogui.keyDown('l')
        pyautogui.keyUp('l')
        time.sleep(0.05)
        i +=1

def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

def train_run(model):
    
    
    
    return 0
