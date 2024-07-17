import tensorflow as tf

model = tf.keras.models.load_model('C:/Users/emili/Downloads/practica2/cnn_model.h5')

model.save('C:/Users/emili/Downloads/practica2/cnn_model.keras', include_optimizer=False)

