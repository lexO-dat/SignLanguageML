from coordinate_normalization import obtenerAngulos
from conditions_predict import condicionalesLetras
import mediapipe as mp
import cv2
import asyncio



async def process_webcam():
    """
    Process frames from webcam with mediapipe and return predicted letter
    """

    word = ""
    prev = None
    previus = None
    counter = 0

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)  # Usar la webcam predeterminada, puedes cambiar el índice si tienes múltiples cámaras

    with mp_hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo obtener el fotograma (fin de la transmisión). Saliendo...")
                break

            frame = cv2.flip(frame, 1)  # Voltear la imagen horizontalmente
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.multi_hand_landmarks:
                continue

            width, height, _ = frame.shape
            angulosid = obtenerAngulos(results, width, height)[0]
            dedos = ()
            # pulgar externo
            if angulosid[5] > 125:
                dedos += (1,)
            else:
                dedos += (0,)
            # pulgar interno
            if angulosid[4] > 150:
                dedos += (1,)
            else:
                dedos += (0,)
            # 4 dedos
            for id in range(0, 4):
                if angulosid[id] > 90:
                    dedos += (1,)
                else:
                    dedos += (0,)
            totalDedos = dedos.count(1)
            data = condicionalesLetras(dedos)

            if previus == None or data != previus:
                previus = data
                counter+=1
            elif data == previus:
                counter+=1

            if counter == 20:
                print("Letra predicha:", data)
                print("-------------------")
                counter = 0


            cv2.imshow('Hand Gesture Recognition', frame)

            # Detener el bucle cuando se presione 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Ejecutar la función para procesar la webcam
asyncio.run(process_webcam())
