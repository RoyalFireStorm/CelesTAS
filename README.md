# CelesTAS

Artificial Intelligences are increasingly appearing in video games, although they tend to be in retro games, which have simple mechanics or few controls. This leaves behind the games of the moment, such as Hollow Knight, Slay the Spire or Celeste. For this reason, a project called CelesTAS has been designed, which allows you to train the game with your favorite games and then allow the program to create new games similar to yours. This allows for uninterrupted gameplay, which can lead to new discoveries in the world of speedrunning or finding new bugs for developers to fix.

## Instalation
* A copy of the Celeste video game.
* The Celeste Studio mod. All the information regarding installation and instructions for using it with the video game can be found in its GitHub repository. (See below)
* Tesseract (version 5.0 or later is recommended).
* Python (version 3.11.5 is recommended)
* The Python libraries we will use:
  * Numpy.
  * Pandas.
  * mk_dir.
  * Keras.
  * Tensorflow.
  * Pytesseract.
  * Pyautogui.
* Nvidia Cuda (compatible version with the graphics card in use and Tensorflow). See https://www.tensorflow.org/install/gpu?hl=en-419#software_requirements

## Configuration in Celeste

1. Open Celeste and go to the main menu.

2. Select "Mod Options."

3. In the "Celeste TAS" section, make the following configurations:

   - Ensure that all "HitBoxes" options are enabled except for "Simplified hitboxes." This will add hitboxes to interactive objects.

   - "Simplified graphics" should be enabled, and any options that allow hiding should be set to "hide." Leave the rest at their default settings.

   - In "Info HUD," deactivate all options except "GameInfo," "TAS input," and "Subpixel Indicator." "WatchEntity" and "Custom Info" should be turned off.

   - All "RoundValues" options should be set to 2.

   - In "More Options," disable "Center Camera" and "Restore All Settings when TAS Stop." Optionally, you can enable "Launch Studio at Boot" to open Celeste Studio when starting Celeste for convenience.

4. In "HotKeys," make the following modifications in "Keyboard Config":

   - Start/Stop: o
   - Pause/Resume: p
   - Frame Advance: l

## Additional Configuration

In addition to the Celeste settings, you need to modify the `config.txt` file in the project folder with the following variables:

- `tesseractLocation`: Path to `tesseract.exe`, for example: 'C:\Program Files\Tesseract-OCR\tesseract.exe.'

- `endX1`: Smallest X-axis value of the completion rectangle.

- `endX2`: Largest X-axis value of the completion rectangle.

- `endY1`: Smallest Y-axis value of the completion rectangle.

- `endY2`: Largest Y-axis value of the completion rectangle.

- `frames`: Maximum number of frames for execution, valid for both training and prediction.

- `lossFunction`: Loss function to use, must be valid for Tensorflow. Default is "Cosine Similarity."

- `optimizerFunction`: Optimization function to use, must be valid for Tensorflow. Default is "Adam."

- `epochs`: Number of training cycles to perform. Default is 20 training cycles.

- `batchSize`: Batch size to use. Default is 50.

Make sure to adjust these variables according to your needs before running Celeste TAS. Enjoy the game!


## Launching Commands

To launch the training phase of the application, open a terminal, navigate to the project's root folder, and execute the following command:

```CMD
python train.py {name_model} {-c}
```
Where name_model will be the name of the desired model, and -c is an option that forces Tensorflow to use the computer's CPU instead of the GPU. This latter option is optional, and by default, it is disabled.

If you want to initiate the prediction phase, use the following command in the terminal:

```CMD
python playGame.py {model} {name}
```
Where model is the name of the already trained model, and name is the name of the specific race you are going to participate in. The name is solely used for saving race information.

## Reference Projects
- [CelesteTAS](https://github.com/EverestAPI/CelesteTAS-EverestInterop) - This proyect is a tool for the proyect.

