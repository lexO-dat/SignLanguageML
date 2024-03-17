#Esta version del programa usa un 30% menos de cpu en promedio a comparacion de la version test
import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pygame

# Inicializar pygame para reproducir sonidos
pygame.mixer.init()

# Cargar el modelo TensorFlow Lite para la clasificación de gestos
interpreter = tf.lite.Interpreter(model_path="Model/keras_model.tflite")
interpreter.allocate_tensors()

# Obtener los detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# URL de la cámara
esp32_url = "http://192.168.4.2/"

# Inicializar la captura de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ancho de 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto de 480

# Inicializar el detector de manos de cvzone
detector = HandDetector(maxHands=1)

# Definir el margen alrededor de la mano detectada
offset = 20

# Tamaño de imagen esperado por el modelo
imgSize = 224

#Letra anterior, para el conteo de fotogramas
previous_letter = None

# Lista de etiquetas del modelo
labels = ["A", "B", "C"]

# Definir sonidos para cada letra
sounds = {
    "A": "A.wav",
    "B": "B.wav",
    "C": "C.wav"
}

# Función para reproducir el sonido correspondiente a una letra
def play_sound(letter):
    if letter in sounds:
        pygame.mixer.music.load(sounds[letter])
        pygame.mixer.music.play()

# Bucle principal
while True:
    success, img = cap.read()  # Leer un fotograma de la cámara
    imgOutput = img.copy()

    # Detectar manos en la imagen
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Coordenadas del rectángulo que enmarca la mano

        # Recortar la región de interés (ROI) alrededor de la mano
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Verificar si la imagen recortada tiene dimensiones válidas
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            # Ajustar el tamaño de la imagen recortada para que coincida con el modelo
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Preparar los datos de entrada para la inferencia
            input_data = np.expand_dims(imgWhite.astype(np.float32) / 255.0, axis=0)

            # Realizar la inferencia
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Obtener los resultados de la inferencia
            output_data = interpreter.get_tensor(output_details[0]['index'])
            index = np.argmax(output_data)
            current_letter = labels[index]

            # Si la letra actual es la misma que la anterior, incrementar el contador
            if current_letter == previous_letter:
                counter += 1
            else:
                previous_letter = current_letter
                counter = 1

            # Si la letra ha sido detectada durante 2 segundos, imprimir la letra y reproducir el sonido
            if counter >= 20:  # 20 cuadros a 10 fps = 2 segundos
                print("LA LETRA ES:", current_letter)
                play_sound(current_letter)
                letter_time = time.time()
                counter = 0  # Reiniciar el contador

        else:
            print("imgCrop tiene dimensiones inválidas:", imgCrop.shape)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)