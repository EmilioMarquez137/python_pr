from mpi4py import MPI
import numpy as np

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

def main():
    comm= MPI.COMM_WORLD
    rank= comm.Get_rank()
    size= comm.Get_size()

    if rank== 0:
        total_num= 100000
        valores= np.random.random(total_num)
        cargas= nivelar_cargas(valores, size)
    else:
        cargas= None

    l= comm.scatter(cargas, root=0)
    m= np.sum(l)
    r= len(l)
    h= comm.gather(m, root=0)
    global_counts= comm.gather(r, root=0)

    if rank== 0:
        total_sum= sum(h)
        total_count= sum(global_counts)
        media= total_sum/ total_count
        print(f"Media global: {media}")

if __name__ == "__main__":
    main()