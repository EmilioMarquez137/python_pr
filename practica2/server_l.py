"""
Primero se establecen la dirección y el puerto.
Se carga el dataset de iris y se hace la división
en entrenamiento y prueba para luego hacer el modelo KNN
Recibe e array de características para devolver
la predicción
"""

import socket
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

dir= 'localhost'
port= 53580

iris= load_iris()
x_train, x_test, y_train, y_test= train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)

print("Modelo KNN entrenado")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((dir, port))
    s.listen()
    print(f"Servidor escuchando en {dir}:{port}")
    conn, addr= s.accept()
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            lm= pickle.loads(data)
            resultado= knn.predict([lm])
            flor= iris.target_names[resultado[0]]
            conn.sendall(pickle.dumps(flor))