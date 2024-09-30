#####################################################
#### Autor: Guido Glorioso
#### Descripcion: Funciones para procesamiento de imagen deteccion 
####              y confeccion de grilla.
####
#####################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: documentar bien las funciones
# TODO: detectar errores y prevenir
# TODO: mutar a clase

def check_color(color:str):
    """Chequeo de color

    Args:
        color (str): Color a chequear

    Raises:
        ValueError: en caso de color erroneo

    Returns:
        bool: True en exito
    """
    
    colores = ["verde" , "amarillo", "negro"]

    if color in colores:
        return True
    
    raise ValueError(f"El color '{color}' es inválido.")
    return False

def filtro_color(img : np.array, color_filtro : str) -> np.array: 
    """Filtrado por color. Posibles filtros ["verde","amarillo","negro"] 

    Args:
        img (np.array): matriz imagen formato RGB
        color_filtro (str): Color a filtrar

    Returns:
        np.array: Imagen filtrada
    """
    if not check_color(color_filtro): return 0

    _img = img.copy()

    # Convierto a HSV para filtrar mas facil
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2HSV)


    # limites de deteccion
    if color_filtro == "negro":
        limite_superior= np.array([200, 130,100]) # [200, 130,100] # Detecto Negro 
        limite_inferior = np.array([0, 0, 0])     # [0, 0, 0]

    elif color_filtro == "amarillo":
        limite_superior= np.array([60, 250, 255]) # [60, 250, 255]Detecto amarillo 
        limite_inferior = np.array([5, 120, 190]) # [5, 120, 190]

    elif color_filtro == "verde":
        limite_superior= np.array([90, 250,200])  # [90, 250,200] Detecto verde 
        limite_inferior = np.array([60, 55, 30])  # [60, 55, 30]
    
    # Establezco color blanco a los objetos filtros obtenidos
    _or = np.ones(shape=_img.shape,dtype= np.uint8) * 255
    
    
    #Realizo mascara de color
    _mascara= cv2.inRange(_img, limite_inferior, limite_superior)
    _img = cv2.cvtColor(_img, cv2.COLOR_HSV2RGB)
    _detectado = cv2.bitwise_or(_img, _or, mask=_mascara)

    #Invierto colores. Objetos negros, fondo blanco
    _detectado = cv2.bitwise_not(_detectado)

    return _detectado

def detectar_referencia_px(img : np.array, color_ref : str = "verde",ref :float = 5) -> tuple:
    """Detecta lineas de color "color_ref" e indicando el valor previo de las lineas (ref) calcula la medicion de cada pixel.\n
    Devuelve ademas la imagen orientada segun la referencia.\n
    NOTA:\n
    Es necesario que las lineas tengan la misma distancia y una encima de la otra en forma paralela (cualquier distancia entre lineas). En caso tipico 3 lineas de 5cm funciona correcto .

    Args:
        img (np.array): imagen
        color_ref (str, optional): Color a detectar como referencia. Defaults to "verde".
        ref (float, optional): largo de linea en cm. Defaults to 5.

    Returns:
        tuple: (unidades pixel, imagen rotada)
    """

    _img = img.copy()

    # Aplico filtro referencia
    _img2 = filtro_color(_img, color_ref)
   
    # Obtengo bordes de lineas
    _img2 = cv2.Canny(_img2,  threshold1=50,  threshold2=60, apertureSize=3)

    # Detecto lineas
    _lineas = cv2.HoughLinesP(_img2, rho=1, theta=np.pi/180, threshold=200, minLineLength=180, maxLineGap=30)
    
    if len(_lineas) == 0:
    
        raise ValueError(f"La imagen no posee referencia adecuada")
    
    else:
        # Promedio y obtengo valor medio de los angulos detectados
        _angulos = np.array([np.degrees(np.arctan((y2 - y1)/(x2 - x1))) for linea in _lineas for x1, y1, x2, y2 in linea]) 
        _angulo_rotacion = _angulos.mean()

        # Roto imagen original con referencia en las lineas detectadas
        _alto, _ancho, _ = _img.shape
        _centro = (_ancho // 2, _alto // 2)
        _matriz_rotacion = cv2.getRotationMatrix2D(_centro, _angulo_rotacion, 1.0)
        imagen_rotada = cv2.warpAffine(_img, _matriz_rotacion, (_ancho, _alto), flags=cv2.INTER_CUBIC,borderValue=(255, 255, 255))  # Relleno con pixeles blancos
        
    # Repito proceso de deteccion con lineas rotadas (mejora la deteccion)
    _img2 = filtro_color(imagen_rotada, color_ref)
   
    _img2 = cv2.Canny(_img2,  threshold1=10,  threshold2=30, apertureSize=3)

    _lineas = cv2.HoughLinesP(_img2, rho=0.5, theta=np.pi/180, threshold=100, minLineLength=200, maxLineGap=100)

    # Guardo la linea mas larga que se puede formar.

    # TODO: cambiar el metodo de deteccion por uno mas confiable 
    
    _x1 =  np.min(_lineas[:, 0, 0])  # Columna de X1
    _x2 = np.max(_lineas[:, 0, 2])  # Columna de X2

    _largos = np.array([np.sqrt((_x2 - _x1)**2 + (y2 - y1)**2) for linea in _lineas for _, y1, _, y2 in linea]) 
    
    # Calculo medicion de un pixel
    unidad_px = ref / _largos.mean()

    return unidad_px,imagen_rotada

def buscar_centro(imagen:np.array ,color_ref:str = "amarillo") -> np.array:

    filtro_ =  filtro_color(imagen,color_ref)

    coordenadas_negra = np.argwhere(cv2.bitwise_not(filtro_amarillo))
    promedio = np.round(coordenadas_negra.mean(axis=0))
    return promedio[0:2][::-1]

def trasladar_imagen_referencia(imagen, coord_ref, coord_dest, dim_out=0):
    # Cargar la imagen original
    _imagen = imagen.copy()
    
    # Obtener las dimensiones de la imagen
   
    alto, ancho = _imagen.shape[:2]
        
    _desp_x =   coord_dest[0] - coord_ref[0]
    _desp_y =   coord_dest[1]  - coord_ref[1]
    
    # Crear una matriz de transformación (traslación)
    M = np.float32([[1, 0, _desp_x], [0, 1, _desp_y]])
    
    # Aplicar la transformación de traslación
    imagen_trasladada = cv2.warpAffine(imagen, M, (ancho, alto), borderValue=(255, 255, 255)) # Rellena con blanco
    if dim_out > 0:
        imagen_trasladada =  imagen_trasladada[0:dim_out,0:dim_out]
    return imagen_trasladada

def resize_y_threshold(imagen, nuevo_ancho, nuevo_alto, threshold=127):
    # Hacer un resize de la imagen a 200x200 usando el método de INTER_AREA (adecuado para reducir)
    imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)
    imagen_redimensionada = cv2.bitwise_not(imagen_redimensionada)
    # Aplicar el threshold: si el valor promedio del píxel supera el threshold, se pone negro (0)
    # Sino, se pone blanco (255)
    _, imagen_binarizada = cv2.threshold(imagen_redimensionada, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return imagen_binarizada

def crop_img(img,centro, pxl_dim):
    
    dim = [pxl_dim[0],pxl_dim[1],3]

    _ret = np.ones(dim,dtype=np.uint8)*255

    pxl_x_inicial =  max(0, int(centro[0] - pxl_dim[0] // 2))
    pxl_x_final   =  min(img.shape[0], int(centro[0] + pxl_dim[0]//2))
    pxl_y_final   =  int(centro[1]) + 50
    pxl_y_inicial =  max(0,int(centro[1] - pxl_dim[1] + 50))
    
    _img_crop = img[pxl_y_inicial:pxl_y_final,pxl_x_inicial:pxl_x_final,:]

    x,y = _img_crop.shape[0:2]
    _ret[:x,:y,:] = _img_crop
 
    return _ret


def get_matriz(ruta_img):

    # Levanto la imagen y la proceso en RGB
    _img = cv2.imread(ruta_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)

    # Detecto la referencia, obtengo medida de px y roto imagen segun referencia
    try:
        _px_dim, _img_rot = detectar_referencia_px(_img,"verde",ref=5) # Ref: medida en cm de la linea de referencia
    except ValueError as e:
        print(e)
        return 0
    
    # Detecto referencia de centro
    _centro = buscar_centro(_img_rot)

    # Defino rejilla
    # Parametros rejilla:
    dim_cuadrado=0.1 # Cm lado
    dim_rejilla = 20 # Cm por lado

    # Calculo la cantidad de pixeles para la cuadricula.
    _cant_cuadrados = int(dim_rejilla / dim_cuadrado)

    _cant_pxl_cuadrado = dim_cuadrado/ _px_dim

    _cant_pxl_rejilla = int(_cant_pxl_cuadrado * _cant_cuadrados)

    # Recorto la imagen para que tenga el tamaño de pixeles deseado (Solo queda una superficie de (dim_rejilla X dim_rejilla) )
    _img = crop_img(_img_rot,_centro,[_cant_pxl_rejilla,_cant_pxl_rejilla])
    
    # Determino el punto donde quiero mover la referencia
    _new_centro = [_img.shape[0]//2, _img.shape[1]]
    # Busco el centro de la imagen
    _centro = buscar_centro(_img)
    
    # Me quedo solo con las lineas negras que son las distancias que quiero medir.
    _img = filtro_color(_img,"negro")
    
    # Traslado la imagen desde el punto de referencia al punto deseado en la imagen
    _img = trasladar_imagen_referencia(_img, _centro, _new_centro)

    # Promedio los cuadrados de la rejilla para que solo tengan un PXL (bajo resolucion)
    _resultado = resize_y_threshold(_img,200,200,threshold=50)

    return _resultado


def generar_cuadrilla(imagen, separacion_px, color=(255, 0, 0)):
    # Crear una imagen blanca del tamaño especificado
    _imagen = imagen.copy()
    ancho,alto = _img.shape[0:2]
    # Dibujar líneas horizontales y verticales con la separación deseada
    for x in range(0, ancho, separacion_px):
        cv2.line(_imagen, (x, 0), (x, alto), color, 2)  # Líneas verticales

    for y in range(0, alto, separacion_px):
        cv2.line(_imagen, (0, y), (ancho, y), color, 2)  # Líneas horizontales

    return _imagen
