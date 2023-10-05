import os
import re
import sys
import pyautogui
import time
import numpy as np
import psutil
import pandas as pd
from pytesseract import pytesseract

keep_going = True
#Aspect ratio 16:9. We are looking for nHD standard resolution
INPUT_WIDTH = 640
INPUT_HEIGHT = 360
INPUT_CHANNELS = 3
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
def parsedStatuses(inputs):
    sol = [0] * 8

    for status in inputs:
        if(status == 'Wall-R'):
            sol[0] = 1
        if(status == 'Wall-L'):
            sol[1] = 1
        if(status == 'CanDash'):
            sol[2] = 1
        if(status == 'Ground'):
            sol[3] = 1
        if(status == 'Dead'):
            sol[4] = 1
        if(status == 'Coyote'):
            sol[5] = 1
        if(status == 'NoControl'):
            sol[6] = 1
        if(status == 'Frozen'):
            sol[7] = 1
    return sol


def screenshot():
    time.sleep(0.02)
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
            state = 'StNormal'
    if(state=="StIntroRespawn"):
        stateNum = 0
    elif(state=="StDash"):
        stateNum = 3
    elif(state=="StClimb"):
        stateNum = 2
    else:
        stateNum = 1
    
    try:
        status = rowFrame['Statuses'].strip().split(' ')
        statuses = parsedStatuses(status)
    except:
        statuses = [0] * 8

    """ if Frame-1 in info:
        rowPreviousFrame = info.loc[Frame-1]
        if (type(rowPreviousFrame)!=pd.Series):
            rowPreviousFrame = rowPreviousFrame.iloc[0]
        Previousstatuses = rowPreviousFrame['Statuses'].strip().split(' ')
        if 'Dead' in Previousstatuses: statuses.append('Dead') """
    

    return posX, posY, spdX, spdY, stateNum, statuses, inputs

def grabar_juego():
    file_dir = select_celeste_path()[:-11] + f"training.txt"
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
    info = pd.read_csv(file_dir, delimiter='\t',index_col='Frames')
    info = info.drop(columns=['Line','Entities'])
    print(info.size)
    while death == False and end_level == False:
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
                if (2043 <= int(float(posX)) <= 2111) & (40 >= int(float(posY)) + 12 >= 74): #Goal Coordinates (+12 because the coordinates of the player are in their feets)
                    end_level = True
                    print('End Level')
                avanzarframe(2)
                frames = frames + 2
    print('The recording has stopped')

    return data
def avanzarframe(num):
    i=0
    while i<num:
        pyautogui.keyDown('l')
        pyautogui.keyUp('l')
        time.sleep(0.02)
        i +=1

def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

def select_celeste_path():
    pyautogui.FAILSAFE = False
    discs = psutil.disk_partitions()
    question = "In what disc is your Celeste game? "
    text = "Please select: "
    root_paths = []
    for disc in discs:
        root_paths.append(disc.device)
        text = text + "\n " + disc.device[0] + " if you want to search in disc " + disc.device
    question = question + text
    selected_disc = pyautogui.prompt(question, default = discs[0].device[0])
    while(selected_disc != None):
        if selected_disc in root_paths or selected_disc+":\\" in root_paths:
            if(len(selected_disc)== 1):
                selected_disc =  selected_disc + ":\\"
            break
        else:
            error_Message = "Could not find disc " + selected_disc + " in your root directory. Please try again using only the letter or the disc format.\n" + text
            selected_disc = pyautogui.prompt(error_Message, default = discs[0].device[0])
    if(selected_disc == None):
        sys.exit("Shutting down requested by the user.")
    print("Seaching Celeste in disc " + selected_disc + "..." + " This could take some time...")
    dir_results = find_files("Celeste.exe", selected_disc)
    r = re.compile(".*Celeste\\\\Celeste\.exe")
    dir_results = list(filter(r.match, dir_results))
    if(len(dir_results) == 0):
        sys.exit("No Celeste files found for disc " + disc.device)
    return dir_results[0]

def find_files(filename, search_path):
   result = []

# Walking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result

def capture_info(im):
    text = pytesseract.image_to_string(im)
    split1 = text.split('\n')
    print(split1)

    aux = 0
    frame,stateNum = 0, 0
    posX, posY, spdX, spdY = 0.0, 0.0, 0.0, 0.0
    statuses = [0] * 8
    for slice in split1:
        slice = slice.strip()
        if not slice:
            continue
        if "/" in slice:
            if(aux == 0):
                frame = int(slice.split('/', 1)[0])
                continue
            else: continue
        if "," in slice:
            d = re.findall("-?\d+\.\d+", slice)
            if len(d) == 2:
                if(aux == 0):
                    posX, posY = float(d[0]), float(d[1])
                    aux = 1
                elif(aux == 1):
                    spdX, spdY = float(d[0]), float(d[1])
                    aux = -1
                elif(aux == -1):
                    continue
            else:
                continue
            
        if(bool(re.findall("Wall-?R", slice, re.IGNORECASE))):
            statuses[0] = 1
        if(bool(re.findall("Wall-?L", slice, re.IGNORECASE))):
            statuses[1] = 1
        if(bool(re.findall("CanDash", slice, re.IGNORECASE))):
            statuses[2] = 1
        if(bool(re.findall("Ground", slice, re.IGNORECASE))):
            statuses[3] = 1
        if(bool(re.findall("Dead", slice, re.IGNORECASE))):
            statuses[4] = 1
        if(bool(re.findall("Coyote", slice, re.IGNORECASE))):
            statuses[5] = 1
        if(bool(re.findall("NoControl", slice, re.IGNORECASE))):
            statuses[6] = 1
        if(bool(re.findall("Frozen", slice, re.IGNORECASE))):
            statuses[7] = 1
        if(bool(re.findall("StIntroRespawn", slice, re.IGNORECASE))):
            stateNum = 0
        elif(bool(re.findall("StDash", slice, re.IGNORECASE))):
            stateNum = 1
        elif(bool(re.findall("StClimb", slice, re.IGNORECASE))):
            stateNum = 2
        elif(bool(re.findall("StNormal", slice, re.IGNORECASE))):
            stateNum = 3
        
    return frame, posX, posY, spdX, spdY, stateNum, statuses