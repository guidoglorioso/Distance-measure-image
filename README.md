# Proyecto de Procesamiento de Imágenes

## Descripción del Proyecto

Este proyecto tiene como objetivo procesar imágenes para detectar y medir objetos utilizando referencias de color. Se utilizan técnicas de procesamiento de imágenes como la detección de bordes y la transformación de Hough para identificar líneas y calcular distancias. El proyecto está dividido en varias etapas, incluyendo la detección de referencias, la rotación de imágenes y la generación de cuadrículas para la medición.

## Estructura del Proyecto

El proyecto está organizado en los siguientes directorios:
project/ <br>
├── docs/ <br>
│ ├── reports/ <br>
│ ├── specifications/<br> 
│ └── user_manual/ <br>
├── logs/ <br>
├── scripts/ <br>
│   ├── funciones_camara_IP.py <br>
│   ├── funciones_procesamieto.py <br>
│   ├── images/ <br>
│   └── notebook_desarrollo.ipynb <br>
│   <br>
└── README.md<br>


### scripts/

- **funciones_camara_IP.py**: Contiene la clase `IP_CAMERA` para interactuar con la API de DroidCam.
- **funciones_procesamieto.py**: Contiene funciones para el procesamiento de imágenes, detección y confección de cuadrículas.
- **notebook_desarrollo.ipynb**: Notebook de Jupyter para el desarrollo y pruebas de las funciones de procesamiento de imágenes.
- **pruebas_captura_imagen.ipynb**: Notebook de Jupyter para pruebas de captura de imágenes.

## Requisitos

- Python 3.11.2
- OpenCV
- NumPy
- Matplotlib

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tuusuario/proyecto-procesamiento-imagenes.git
    cd proyecto-procesamiento-imagenes
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Ejecuta el notebook [notebook_desarrollo.ipynb](http://_vscodecontentref_/5) para probar las funciones de procesamiento de imágenes.
2. Utiliza [funciones_procesamieto.py](http://_vscodecontentref_/6) para integrar las funciones en tu propio proyecto.

---

# Image Processing Project

## Project Description

This project aims to process images to detect and measure objects using color references. Image processing techniques such as edge detection and Hough transformation are used to identify lines and calculate distances. The project is divided into several stages, including reference detection, image rotation, and grid generation for measurement.

## Project Structure

The project is organized into the following directories:

project/ <br>
├── docs/ <br>
│ ├── reports/ <br>
│ ├── specifications/<br> 
│ └── user_manual/ <br>
├── logs/ <br>
├── scripts/ <br>
│   ├── funciones_camara_IP.py <br>
│   ├── funciones_procesamieto.py <br>
│   ├── images/ <br>
│   └── notebook_desarrollo.ipynb <br>
│   <br>
└── README.md<br>

### scripts/

- **funciones_camara_IP.py**: Contains the [IP_CAMERA](http://_vscodecontentref_/7) class to interact with the DroidCam API.
- **funciones_procesamieto.py**: Contains functions for image processing, detection, and grid generation.
- **notebook_desarrollo.ipynb**: Jupyter notebook for developing and testing image processing functions.
- **pruebas_captura_imagen.ipynb**: Jupyter notebook for image capture tests.

## Requirements

- Python 3.11.2
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/image-processing-project.git
    cd image-processing-project
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the [notebook_desarrollo.ipynb](http://_vscodecontentref_/8) notebook to test the image processing functions.
2. Use [funciones_procesamieto.py](http://_vscodecontentref_/9) to integrate the functions into your own project.