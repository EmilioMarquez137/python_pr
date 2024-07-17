import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

train= "C:/Users/emili/Downloads/brain/Training"
test = "C:/Users/emili/Downloads/brain/Testing"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(28, 28),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=32
)

#Redsita
model = models.Sequential([
    #Simple y compleja
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    #También usamos la técnica de Dropout
    tf.keras.layers.Dropout(0.4),
    #Aplastamos
    layers.Flatten(),
    #capa sencilla con sus funciónes de activación para clasificadores
    layers.Dense(30, activation='relu'),
    layers.Dense(4, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
# Rendimiento

plt.figure(figsize=(12, 6))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = np.argmax(model.predict(validation_dataset), axis=1)
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)

# Calcular matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Imprimir la matriz de confusión con un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Matriz de confusión')
plt.show()