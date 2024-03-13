import cv2
import tensorflow
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pygame
import requests

# Inicializar pygame
pygame.mixer.init()

esp32_url = "http://192.168.0.5:4747/video"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ancho de 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto de 480
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "data/A"
counter = 0
previous_letter = None
letter_time = time.time()
labels = ["A", "B", "C"]

# Definir sonidos para cada letra
sounds = {
    "A": "A.wav",
    "B": "B.wav",
    "C": "C.wav"
}

def play_sound(letter):
    if letter in sounds:
        pygame.mixer.music.load(sounds[letter])
        pygame.mixer.music.play()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            #print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # Si la letra actual es la misma que la anterior, incrementa el contador
        if labels[index] == previous_letter:
            counter += 1
        else:
            previous_letter = labels[index]
            counter = 1

        # Si la letra ha sido detectada durante 2 segundos, imprímela
        if counter >= 20:  # 20 cuadros a 10 fps = 2 segundos
            print("Letra detectada:", labels[index])
            play_sound(labels[index])
            letter_time = time.time()
            counter = 0  # Reiniciar el contador

        # Mostrar cuánto tiempo ha pasado desde la última letra detectada
        elapsed_time = time.time() - letter_time
        cv2.putText(imgOutput, f"Tiempo: {elapsed_time:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)
    #cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)