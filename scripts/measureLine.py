import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('scripts\images\edit.jpg')

# Convertir a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de desenfoque para suavizar la imagen (opcional, ayuda a mejorar la detección)
imagen_suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

# Aplicar una binarización inversa (blanco y negro)
_, imagen_binaria = cv2.threshold(imagen_suavizada, 150, 255, cv2.THRESH_BINARY_INV)

# Detectar las líneas horizontales
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detecta_lineas = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

# Invertir los colores nuevamente para que las líneas sean negras y el fondo blanco
invertida = cv2.bitwise_not(detecta_lineas)

# Restar las líneas detectadas de la imagen original en escala de grises
imagen_sin_renglones = cv2.bitwise_and(imagen_gris, imagen_gris, mask=invertida)

# Mostrar la imagen resultante sin los renglones
cv2.imshow('Imagen sin renglones', imagen_sin_renglones)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen sin los renglones
cv2.imwrite('scripts\images\edit2.jpg', imagen_sin_renglones)

