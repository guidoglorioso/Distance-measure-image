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
    # Verifico si el color es valido
    check_color(color_filtro)

    _img = img.copy()

    # Convierto a HSV para filtrar mas facil
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2HSV)


    # limites de deteccion
    if color_filtro == "negro":
        limite_superior= np.array([90, 90,90]) # [200, 130,100] # Detecto Negro 
        limite_inferior = np.array([0, 0, 0])     # [0, 0, 0]

    elif color_filtro == "amarillo":
        limite_superior= np.array([60, 250, 255]) # [60, 250, 255]Detecto amarillo 
        limite_inferior = np.array([5, 120, 190]) # [5, 120, 190]

        limite_inferior = np.array([15, 140, 70])  # img caso real
        limite_superior = np.array([35, 255, 255]) # img caso real
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
        img (np.array): imagen RGB
        color_ref (str, optional): Color a detectar como referencia. Defaults to "verde".
        ref (float, optional): largo de linea en cm. Defaults to 5.

    Returns:
        tuple: (unidades pixel, imagen RGB rotada)
    """

    _img = img.copy()

    # Aplico filtro referencia
    _img2 = filtro_color(_img, color_ref)


    # Obtengo bordes de lineas
    _img2 = cv2.GaussianBlur(_img2, (5, 5), 0)
    _img2 = cv2.Canny(_img2,  threshold1=30,  threshold2=150, apertureSize=3)

    #plt.title("Referencia detectada")
    #plt.imshow(_img2)
    #plt.show()
    
    # Detecto lineas
    _lineas = cv2.HoughLinesP(_img2, rho=1, theta=np.pi/360, threshold=100, minLineLength=200, maxLineGap=60) # Ajuste img real

    
    if _lineas is None:
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
   
   # Obtengo bordes de lineas
    _img2 = cv2.GaussianBlur(_img2, (5, 5), 0)
    _img2 = cv2.Canny(_img2,  threshold1=30,  threshold2=150, apertureSize=3)

    # Detecto lineas
    _lineas = cv2.HoughLinesP(_img2, rho=1, theta=np.pi/360, threshold=100, minLineLength=200, maxLineGap=60) # Ajuste img real

    # Guardo la linea mas larga que se puede formar.

    # TODO: cambiar el metodo de deteccion por uno mas confiable 

    _largos = np.array([np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for linea in _lineas for x1, y1, x2, y2 in linea])    
    
    
    # Calculo medicion de un pixel

    unidad_px = ref / _largos.max()

    #print(f"Largos : {_largos.max()}")
    #print(f"Unidad de pixel : {unidad_px * 0.8}")
    return unidad_px,imagen_rotada

def buscar_centro(imagen:np.array ,color_ref:str = "amarillo") -> np.array:
    """Detecta un color en una imagen (color_ref) y en base a todos los pixeles de este color obtiene el punto medio de todos.

    Args:
        imagen (np.array): imagen RGB 
        color_ref (str, optional): color a detectar. Defaults to "amarillo".

    Returns:
        np.array: Coordenadas (x,y) del centro del color
    """
    
    _filtro =  filtro_color(imagen,color_ref)

    _coord = np.argwhere(cv2.bitwise_not(_filtro))
    promedio = np.round(_coord.mean(axis=0))

    return promedio[0:2][::-1]

def trasladar_imagen_referencia(imagen:np.array, coord_ref:tuple, coord_dest:tuple)-> np.array:
    """Traslada una imagen respecto a un centro dado (coord_ref) a un segundo centro (coord_dest)

    Args:
        imagen (np.array): imagen RGB a trasladar
        coord_ref (tuple): coordenadas origen
        coord_dest (tuple): coordenadas destino

    Returns:
        np.array: imagen RGB trasladada
    """

    # Cargar la imagen original
    _imagen = imagen.copy()
    
    # Obtener las dimensiones de la imagen
   
    _alto, _ancho = _imagen.shape[:2]

    # Calculo delta de coords
        
    _desp_x =   coord_dest[0] - coord_ref[0]
    _desp_y =   coord_dest[1]  - coord_ref[1]
    
    # Crear una matriz de transformación (traslación)
    _M = np.float32([[1, 0, _desp_x], [0, 1, _desp_y]])
    
    # Aplicar la transformación de traslación
    _imagen_trasladada = cv2.warpAffine(imagen, _M, (_ancho, _alto), borderValue=(255, 255, 255)) # Rellena con blanco
    
    return _imagen_trasladada

def resize_y_threshold(imagen:np.array, nueva_dim : tuple, threshold:int=127)-> np.array:
    """Toma una imagen y la redimenciona la imagen achicando a nueva_dim pixeles (ancho,alto) y binariza segun el umbral seteado. Todo lo que supere el umbral tendra un pixel negro

    Args:
        imagen (np.array): imagen RGB
        nueva_dim (tuple): dimensiones nuevas de la imagen (nuevo_ancho,nuevo_alto)
        threshold (int, optional): valor umbral para detecta como pixel negro (cuanto mayor umbral los pixeles tienen que ser menos negros). Defaults to 127.

    Returns:
        np.array: imagen Grey binarizada.

    """
    _img = imagen.copy()
    
    # Se hace un resize promediando los pixeles vecinos
    _img = cv2.resize(_img, nueva_dim, interpolation=cv2.INTER_AREA)
    
    # Aplicar el threshold: si el valor promedio del píxel supera el threshold, se pone negro (0)
    # Sino, se pone blanco (255)
    _img = cv2.cvtColor(_img,cv2.COLOR_RGB2GRAY)

    _, imagen_binarizada = cv2.threshold(_img, threshold, 255, cv2.THRESH_BINARY)


    return imagen_binarizada

def grey2rgb(img:np.array):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


def crop_img(img:np.array ,centro:tuple, pxl_dim:tuple, offset:int = 50):
    """
    Recorta una imagen dada centrada en un punto específico y con dimensiones determinadas.
    
    Esta función toma una imagen `img` y recorta una sección centrada en las coordenadas 
    `centro`, con dimensiones especificadas por `pxl_dim`. Si el recorte se extiende 
    más allá de los límites de la imagen, la función llena el área sobrante con píxeles blancos.

    Parámetros:
    ----------
    img : np.array
        La imagen de entrada representada como un arreglo de NumPy con formato (alto, ancho, canales).
    
    centro : tuple
        Una tupla (x, y) que define el centro del área a recortar en la imagen.
    
    pxl_dim : tuple
        Una tupla (alto, ancho) que define las dimensiones (en píxeles) del área a recortar.

    offset: int 
        Cantidad de pixeles para desplazar en Y el centro.
    
    Retorna:
    -------
    np.array
        Una nueva imagen del tamaño especificado por `pxl_dim`, con el área recortada centrada
        en las coordenadas `centro`. Si las dimensiones exceden los límites de la imagen original,
        las áreas vacías se rellenan con píxeles blancos.
    """
    
    _dim = [pxl_dim[0],pxl_dim[1],3]

    _ret = np.ones(_dim,dtype=np.uint8)*255

    pxl_x_inicial =  max(0, int(centro[0] - pxl_dim[0] // 2))
    pxl_x_final   =  min(img.shape[0], int(centro[0] + pxl_dim[0]//2))
    pxl_y_final   =  int(centro[1]) + offset
    pxl_y_inicial =  max(0,int(centro[1] - pxl_dim[1] + offset))
    
    _img_crop = img[pxl_y_inicial:pxl_y_final,pxl_x_inicial:pxl_x_final,:]

    x,y = _img_crop.shape[0:2]
    _ret[:x,:y,:] = _img_crop
 
    return _ret


def get_matriz(ruta_img:str,dim_cuadrado:float,dim_rejilla:float)->np.array:
    """Funcion que recibe una ruta de imagen y obtiene de ella una matriz reducida donde cada pixel representa una distancia "dim_cuadrado" 
    y en donde la cantidad total de pixeles depende de la dimension que se desea representar en total "dim_rejilla"\n
    Por defecto se toma como referencia de medida el color "verde", como centro el color "amarillo" y como objeto el color "negro".

    Args:
        ruta_img (str): ruta imagen
        dim_cuadrado (float): medida de cada pixel en cm
        dim_rejilla (float): medida de la imagen en cm

    Returns:
        np.array: imagen formato grey centrada en la referencia y escalada 
    """
    # Levanto la imagen y la proceso en RGB
    _img = cv2.imread(ruta_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    
    # Detecto la referencia, obtengo medida de px y roto imagen segun referencia
    _px_dim, _img_rot = detectar_referencia_px(_img,"verde",ref=5) # Ref: medida en cm de la linea de referencia

    # Detecto referencia de centro
    _centro = buscar_centro(_img_rot)

    # Calculo la cantidad de pixeles para la cuadricula.
    _cant_cuadrados = int(dim_rejilla / dim_cuadrado)

    _cant_pxl_cuadrado = dim_cuadrado/ _px_dim

    _cant_pxl_rejilla = int(_cant_pxl_cuadrado * _cant_cuadrados)

    # Recorto la imagen para que tenga el tamaño de pixeles deseado (Solo queda una superficie de (dim_rejilla X dim_rejilla) )
    _img = crop_img(_img_rot,_centro,[_cant_pxl_rejilla,_cant_pxl_rejilla],offset=100)

    # Determino el punto donde quiero mover la referencia
    _new_centro = [_img.shape[0]//2, _img.shape[1]]
    # Busco el centro de la imagen
    _centro = buscar_centro(_img)
    
    # Me quedo solo con las lineas negras que son las distancias que quiero medir.
    _img = filtro_color(_img,"negro")

    # Traslado la imagen desde el punto de referencia al punto deseado en la imagen
    _img = trasladar_imagen_referencia(_img, _centro, _new_centro)
    
    # Promedio los cuadrados de la rejilla para que solo tengan un PXL (bajo resolucion)
    _matriz_dim = (200,200)
    _resultado = resize_y_threshold(_img,_matriz_dim,threshold=200)
    #print(f"Factor escala : {_px_dim*_img.shape[0]/200}")
    
    # Calculo de valor de escala por 
    _factor_escala = _px_dim*_img.shape[0]/_matriz_dim[0]

    return grey2rgb(_resultado),_factor_escala



def generar_cuadrilla(imagen:np.array, separacion_px:int, color : tuple=(255, 0, 0)):
    """_summary_

    Args:
        imagen (np.array): imagen RGB.
        separacion_px (int): separacion entre lineas
        color (tuple, optional): Color de la cuadrilla. Defaults to (255, 0, 0).

    Returns:
        _type_: imagen con rejilla
    """

    # Crear una imagen blanca del tamaño especificado
    _img = imagen.copy()

    ancho,alto = _img.shape[0:2]
    # Dibujar líneas horizontales y verticales con la separación deseada
    for x in range(0, ancho, separacion_px):
        cv2.line(_img, (x, 0), (x, alto), color, 2)  # Líneas verticales

    for y in range(0, alto, separacion_px):
        cv2.line(_img, (0, y), (ancho, y), color, 2)  # Líneas horizontales

    return _img
