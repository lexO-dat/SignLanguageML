import cv2
import tensorflow as tf
import numpy as np
import math
import time
import pyttsx3
from cvzone.HandTrackingModule import HandDetector

engine = pyttsx3.init()

# Configuración del programa principal
model_paths = ["../Model/model_hg.tflite", "../Model/model_co.tflite"]
labels_sets = [["h", "g"], ["c", "o"]]

interpreters = []
input_details_list = []
output_details_list = []

# Crear intérpretes para cada modelo
for model_path in model_paths:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreters.append(interpreter)
    input_details_list.append(input_details)
    output_details_list.append(output_details)

print("Interpreters:", interpreters)

esp32_url = "http://192.168.174.119/240x240.jpg"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Inicializar detector de manos
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
word = ""
prev_letter = None
previous_letter = None

while True:
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

                predictions = []
                for interpreter, input_details, output_details in zip(interpreters, input_details_list, output_details_list):
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    predictions.append(output_data)

                # Normalizar las predicciones
                normalized_predictions = []
                for prediction in predictions:
                    normalized_prediction = prediction / np.sum(prediction)
                    normalized_predictions.append(normalized_prediction)
                
                print("Normalized predictions:", normalized_predictions)

                # Combine las predicciones normalizadas
                combined_normalized_predictions = np.sum(normalized_predictions, axis=0)

                # Determinar la letra actual
                index = np.argmax(combined_normalized_predictions)
                current_letter = labels_sets[index // 100][index % 100]

                if current_letter == previous_letter:
                    counter += 1
                else:
                    previous_letter = current_letter
                    counter = 1

                if counter >= 5 and current_letter != prev_letter:
                    prev_letter = current_letter
                    print("LA LETRA ES:", current_letter)
                    if current_letter == "next":
                        engine.say(word)
                        engine.runAndWait()
                        word = ""
                    else:
                        word = word + current_letter

                    print(word)
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
