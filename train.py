import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

#Inicilizacion de la webcam (0 es la camara predeterminada del pc)
cap = cv2.VideoCapture(0)

#Se especifica que se detectara como maximo una mano
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
folder = "data/C"
counter = 0

while True:
    success, img = cap.read()
    # Detectar manos en la imagen
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  #Coordenadas del rectángulo que enmarca la mano

        #Se crea una imagen blanca del tamaño deseado
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        #Recortar la región de interés (ROI) alrededor de la mano
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        #Ajustar el tamaño de la imagen recortada para que coincida con el tamaño deseado
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        #Mostrar las imágenes recortada y blanca
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    #Mostrar la imagen original con las manos detectadas
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        #Se guarda la imagen blanca en la carpeta especificada
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    elif key == ord("q"):
        break