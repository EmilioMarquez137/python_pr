"""
Del lado del cliente se envia una lista de 5 características
de la flor para recibir del servidor la predicción de qué flor
es esa característica
"""
import socket
import pickle

caracteristicas= [1.7,3.8,0.7,0.5]

dir= 'localhost'
port= 53580

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((dir, port))

    s.sendall(pickle.dumps(caracteristicas))
    lm= s.recv(1024)

#res= pickle.loads(lm)
flor= pickle.loads(lm)
print(f"Predicción: {flor}")

