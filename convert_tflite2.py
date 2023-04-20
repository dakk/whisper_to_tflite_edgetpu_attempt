import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('whisper-int8.tflite')

# Disable dynamic tensors
tf.config.run_functions_eagerly = False

# Save the model again with disabled dynamic tensors
loaded_model.save('saved')
