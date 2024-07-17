import os
import random


def delete_random_images(folder_path, num_images_to_delete):
    # Obtener la lista de archivos en la carpeta
    file_list = os.listdir(folder_path)

    # Elegir al azar la cantidad especificada de imágenes para eliminar
    files_to_delete = random.sample(file_list, num_images_to_delete)

    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")


# Ejemplo de uso:
folder_path = 'C:/Users/emili/Downloads/datasets/people'  # Reemplaza con la ruta de tu carpeta
num_images_to_delete = 25000  # Cambia según la cantidad de imágenes que quieras eliminar

delete_random_images(folder_path, num_images_to_delete)

