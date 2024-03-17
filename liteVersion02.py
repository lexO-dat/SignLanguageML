import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)  # Reducir tamaño de imagen
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
labels = ["A", "B", "C"]

interpreter = tf.lite.Interpreter(model_path="Model/keras_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

previous_letter = None
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
            input_data = np.expand_dims(imgResize.astype(np.float32) / 255.0, axis=0)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            index = np.argmax(output_data)
            current_letter = labels[index]

            if current_letter == previous_letter:
                counter += 1
            else:
                previous_letter = current_letter
                counter = 1

            if counter >= 10:
                print("LA LETRA ES:", current_letter)
                letter_time = time.time()
                counter = 0

        else:
            print("imgCrop tiene dimensiones inválidas:", imgCrop.shape)

    cv2.waitKey(50)  # Reducir la frecuencia de captura
