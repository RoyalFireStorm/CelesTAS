import argparse
import os
import sys
from mkdir_p import mkdir_p
import time
import pyautogui
import numpy as np
from train import final_model
from functions import avanzarframe, capture_info, prepare_image, screenshot, parseConfig

comands = ['R','L','U','D','J','X','Z','G','S']

def introTas(name, model_name, level):
    parseConfig("level")
    pyautogui.write(f'console load {level}')
    pyautogui.press('enter')
    pyautogui.write('1')
    pyautogui.press('enter')
    pyautogui.write(f'StartExportGameInfo run_{name}_{model_name}.txt')
    pyautogui.press('enter')
    pyautogui.write('2')
    pyautogui.press('enter')
    pyautogui.write('90')
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
    mkdir_p("weights")
    weights_file = f"weights/{model_name}.hdf5"
    model = final_model()
    if os.path.isfile(weights_file) == False:
        sys.exit("The model called " + model_name + " does not exists.")    
    model.load_weights(weights_file)
    pyautogui.alert('Remenber to set Celeste Studio in the front with a new clear documenta and maximaze the window. It is necesary too to have Celeste to be the previous window so we can tab it. Press OK to start.')
    time.sleep(5)
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
    avanzarframe(93 - frames)
    alttab()
    time.sleep(0.3)
    try:
        framesLimit = int(parseConfig("frames", 999999))
        endX1 = int(parseConfig("endX1"))
        endX2 = int(parseConfig("endX2"))
        endY1 = int(parseConfig("endY1"))
        endY2 = int(parseConfig("endY2"))
    except:
        sys.exit("The config value of the frames or the end points are not integers. Check it and try again.")
    

    while death == False and end_level == False and frames <= framesLimit:
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
            if (endX1 <= int(float(posX)) <= endX2) & (endY1 >= int(float(posY)) + 12 >= endY2):
                end_level = True
            avanzarframe(2)
            frames = frames + 2
    alttab()
    pyautogui.write('EndExportGameInfo')
    alttab()
    avanzarframe(10)
    pyautogui.keyDown('p')
    pyautogui.keyUp('p')
    print("The run is over.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('name')
    args = parser.parse_args()

    playTime(args.model, args.name)