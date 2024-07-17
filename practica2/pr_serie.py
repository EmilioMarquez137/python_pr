from multiprocessing import Pool
import multiprocess

def nivelar_cargas(valores, num_procesadores):
    carga_promedio= len(valores)// num_procesadores
    cargas= []
    ini= 0

    for i in range(num_procesadores):
        if i < len(valores) % num_procesadores:
            fin= ini+ carga_promedio+ 1
        else:
            fin= ini+ carga_promedio

        cargas.append(valores[ini:fin])
        ini= fin

    return cargas

def sumar(serie,lock):
    suma=sum(serie)
    lock.acquire()
    print(suma)
    lock.release()
    return suma

def suma_simple(serie):
  ss= sum(serie)
  return ss

pasos= 1000
procesadores=6
serie= []

#Generar  la serie
for i in range(0, (pasos*2)+1 , 2):
    fraccion = 1 / (i + 1)
    serie.append(fraccion)

#print(serie)
#print(len(serie))
#print(sum(serie))

def exclu(serie, procesadores):
    lock = multiprocess.Lock()
    carga = nivelar_cargas(serie, procesadores)
    procesos = []


    for i in range(procesadores):
        proceso = multiprocess.Process(target=sumar, args=(carga[i], lock))
        procesos.append(proceso)
        proceso.start()



if __name__ == "__main__":
    pasos = 1000
    procesadores = 4
    serie = []

    for i in range(0, pasos * 2, 2):
        fraccion = 1 / (i + 1)
        serie.append(fraccion)

    exclu(serie, procesadores)
    #print(sum(f))