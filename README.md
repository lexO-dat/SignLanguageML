# SignLanguageML
Este proyecto consiste en un prototipo funcional para el reconocimiento de letras en en lenguaje de señas y en tiempo real utilizando una Raspberry Pi, tensorflow y un ESP32.

1. **Entrenamiento del Modelo**: `train.py`
2. **Transformacion del modelo** `tolite.py`
3. **Programa de Detección de Gestos**: `liteVersion.py`

## Entrenamiento del Modelo

El script `train.py` se utiliza para entrenar el modelo de reconocimiento de gestos. A continuación, se detalla su funcionamiento:

### Uso

1. Ejecute el script `train.py` en su entorno de Python preferido.
2. Se abrirá una ventana de la cámara en tiempo real.
3. Coloque la mano frente a la cámara y realice los gestos deseados (por ejemplo, "A", "B", "C").
4. Presione la tecla `s` para capturar y guardar una imagen del gesto actual.
5. El script guardará las imágenes en la carpeta `data/A`, `data/B` y `data/C` respectivamente, según el gesto realizado.
6. Repita este proceso para capturar múltiples muestras de cada gesto.
7. Una vez que haya capturado suficientes muestras, puede utilizar estas imágenes para entrenar un modelo de reconocimiento de gestos.
8. Usando la web de Teachable machines se entrena el modelo de Keras/Tensorflow entregando las imagenes capturadas.

## Transformacion del Modelo

El modelo entrenado con TensorFlow se convierte a un formato más ligero y eficiente llamado TensorFlow Lite. Esto se hace con el script `tolite.py`, el cual toma el modelo entrenado y crea una versión optimizada para su uso en dispositivos embebidos como la Raspberry Pi.

## Uso
Para convertir el modelo a TensorFlow Lite: python tolite.py

## Programa de Detección de Gestos

El script `liteVersion.py` utiliza el modelo entrenado para detectar gestos de manos en tiempo real. A continuación, se detalla su funcionamiento:

### Uso

1. Ejecute el script `liteVersion.py` después de haber entrenado el modelo.
2. Se abrirá una ventana de la cámara en tiempo real.
3. El programa detectará su mano y el gesto que esté realizando.
4. En la esquina superior izquierda de la ventana, se mostrará el gesto detectado.
5. Si se detecta el mismo gesto durante 2 segundos, se reproducirá un sonido correspondiente al gesto.

## Requisitos

- Python 3.8.10
- OpenCV
- TensorFlow
- numpy
- cvzone
- pygame

## Estructura de Carpetas
├── data
│ ├── A
│ ├── B
| ├── C
│ └── . . .
├── Model
│ ├── keras_model.h5
| ├── labels.txt
│ └── keras_model.tflite
├── liteVersion.py
├── test.py
├── toLite.py
├── train.py
└── README.md

