# CelesTAS

Las Inteligencias Artificiales están cada vez apareciendo más en los videojuegos, aunqué suelen ser en los retro, que tiene mecánicas simples o pocos controles. Esto deja atrás a los juegos del momento, como puede ser Hollow Knight, Slay the Spire o Celeste. Por ello, se ha diseñado un proyecto llamado CelesTAS, que permite entrenar al juego con tus partidas favoritas para después permitir al programa de crear nuevas partidas similares a las tuyas. Esto permite jugar ininterrupidamente, lo que puede dar a lugar a nuevos descubrimientos en el mundo del speedrunning o encontrar nuevos errores para que los desarrolladores lo arreglen.

## Instalación
 Para el correcto funcionamiento de la aplicación es necesario:
    - Una copia del videojuego Celeste.
    - El mod Celeste Studio. Toda la información sobre la instalación e instrucciones para usarlo con el videojuego la podemos encontrar en su repositorio de Github. https://github.com/EverestAPI/CelesteTAS-EverestInterop
    - Tesseract (recomendable la versión 5.0 en adelante)
    - Python (recomendable la versión 3.11.5)
    - Las librerías que usaremos en Python:
        - Numpy.
        - Pandas.
        - mk_dir.
        - Keras.
        - Tensorflow.
        - Pytesseract.
        - Pyautogui.
    - Nvidia Cuda (versión compatible con la gráfica en uso y Tensorflow). Véase https://www.tensorflow.org/install/gpu?hl=es-419#software_requirements


## Lanzamiento de la aplicación

Para lanzar la fase de entrenamiento la aplicacíon, abriremos una terminal, accederemos a la carpeta raiz del proyecto y ejecutaremos:
```CMD
python train.py {name_model} {-c}
```

Donde name model ser a el nombre del modelo que deseemos y -c ser a una opcion que obligara a Tensorflow a usar la CPU del ordenador en vez de la GPU. Esta ́ultima opcion es opcional y por defecto esta desactivada.

En caso de querer lanzar la fase de prediccion, usaremos el siguiente comando en la terminal:
```CMD
python playGame.py {model} {name}
```

Donde model sera el nombre que tiene el modelo ya entrenado y name ser ́a el nombre de la carrera en haremos en cuestion. El nombre solo se dispone para el guardado de la informacion de la carrera.

