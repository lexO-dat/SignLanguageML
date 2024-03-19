import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


# Configuración del programa principal
interpreter = tf.lite.Interpreter(model_path="Model/keras_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

esp32_url = "http://192.168.174.119/240x240.jpg"
cap = cv2.VideoCapture(esp32_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
previous_letter = None
labels = ["A", "B", "C"]
sounds = {
    "A": "A.wav",
    "B": "B.wav",
    "C": "C.wav"
}

# Función para reproducir sonidos
# def play_sound(letter):
#     if letter in sounds:
#         pygame.mixer.music.load(sounds[letter])
#         pygame.mixer.music.play()

while True:
    cap.open(esp32_url)

    success, img = cap.read()

    if success:
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

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

                input_data = np.expand_dims(imgWhite.astype(np.float32) / 255.0, axis=0)

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

                if counter >= 5:  # 20 frames at 10 fps = 2 seconds
                    print("LETTER IS:", current_letter)
                    #play_sound(current_letter)
                    letter_time = time.time()
                    counter = 0

            else:
                print("imgCrop has invalid dimensions:", imgCrop.shape)

        cv2.imshow("Image", imgOutput)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Error al leer el fotograma de la cámara.")
        break

cap.release()
cv2.destroyAllWindows()
