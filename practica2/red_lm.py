import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools

image_width, image_height = 720, 480
batch_size = 32
epochs = 10
num_classes = 3
input_shape = (image_width, image_height, 3)

dataset_dir = 'C:/Users/emili/Downloads/datasets'


def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            try:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_width, image_height))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image file {img_path}: {str(e)}")

    return np.array(images), np.array(labels), class_names


# Cargar datos
images, labels, class_names = load_images_from_folder(dataset_dir)
print(f'Loaded {len(images)} images from {len(class_names)} classes.')


def build_model(input_shape):
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Construir y compilar el modelo
model = build_model(input_shape)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callback para imprimir accuracy y loss al final de cada época
class PrintAccuracyLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f} - Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")


# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[PrintAccuracyLossCallback()]
)

# Guardar el modelo
model.save('cnn_model.keras', include_optimizer=False)

# Graficar la precisión y la pérdida
def plot_metrics(history, metric):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history[metric], label='Train')
    plt.plot(history.history[f'val_{metric}'], label='Validation')
    plt.title(f'{metric.capitalize()} per epoch')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

plot_metrics(history, 'accuracy')
plot_metrics(history, 'loss')

# Evaluar el modelo
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de confusión y reporte de clasificación
conf_matrix = confusion_matrix(y_test, y_pred_classes)
class_report = classification_report(y_test, y_pred_classes, target_names=class_names)

print("Confusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Graficar la matriz de confusión
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(conf_matrix, class_names)