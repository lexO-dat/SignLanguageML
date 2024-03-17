import tensorflow as tf

# Cargar el modelo Keras
model = tf.keras.models.load_model("Model/keras_model.h5")

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TensorFlow Lite
with open("Model/keras_model.tflite", "wb") as f:
    f.write(tflite_model)
