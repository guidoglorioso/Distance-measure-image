#####################################################
#### Autor: Guido Glorioso
#### Descripcion: Clase para interaccion con API droidcam
####
#####################################################


import cv2
import urllib.request as urlreq


class IP_CAMERA:
    # Configuración inicial
    PROTOCOL = 'http'
    PORT = '4747'
    DIRNAME = 'video'
    SIZE320x240 = '?320X240'
    SIZE640x480 = '?640X480'  # Por defecto
    SIZE1280x720 = '?1280X720'  # Se requiere licencia PRO en DroidCAM
    SIZE1920x1080 = '?1920X1080'  # Se requiere licencia PRO en DroidCAM
    
    # Comandos útiles
    def __init__(self, ip_address, resolution=SIZE640x480):
        """Clase IP_CAMERA para interactuar con API de droidCam

        Args:
            ip_address (str): IP de la camara (ej 192.168.0.100)
            resolution (str, optional): Resolucion de la imagen. Defaults to SIZE640x480.
        """
        self.ip_address = ip_address
        self.resolution = resolution
        self.cameraURI = f"{self.PROTOCOL}://{self.ip_address}:{self.PORT}/"
        self.cap = None
        self.angle = 0

        # Comandos de la cámara
        self.autoFocus = self.cameraURI + 'cam/1/af'
        self.zoomIn = self.cameraURI + 'cam/1/zoomin'
        self.zoomOut = self.cameraURI + 'cam/1/zoomout'
        self.toggleLED = self.cameraURI + 'cam/1/led_toggle'
        self.fpsRestriction = self.cameraURI + 'cam/1/fpslimit'
        self.getBattery = self.cameraURI + 'battery'

    def cmdSender(self, cmd):
        """Funcion para enviar comandos a la API

        Args:
            cmd (command): Comandos definidos en la clase.

        Returns:
            str: Respuesta de host
        """
        # Función para enviar comandos a la cámara
        ret = ''
        try:
            fp = urlreq.urlopen(cmd)
            ret = fp.read().decode("utf8")
            fp.close()
        except Exception as e:
            print(f"Error al enviar comando: {e}")
        return ret

    def init_camera(self):
        """Inicializacion de camara

        """
        # Inicializa la cámara IP con la URI correcta
        print(f"Conectando a la cámara IP en {self.cameraURI + self.DIRNAME + self.resolution}")
        self.cap = cv2.VideoCapture(self.cameraURI + self.DIRNAME + self.resolution)
        
        if not self.cap.isOpened():
            print("No se pudo conectar a la cámara.")
        else:
            print("Cámara conectada con éxito.")

    def get_img(self):
        """Devuelve imagen obtenida en la camara.

        Returns:
            np.array: Matriz de imagen RGB (resolucion variable)
        """
        # Captura una imagen de la cámara IP
        if self.cap is None or not self.cap.isOpened():
            print("La cámara no está conectada.")
            return None

        ret, frame = self.cap.read()
        if ret:
            # Rotar imagen segun configuracion
            frame = self._rotate_image(frame,self.angle)
            
            # Configuro formato RGB
            return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        else:
            print("No se pudo obtener la imagen.")
            return None
        

    def set_rotation_angle(self, angle):
        """Setea el angulo de rotacion

        Args:
            angle (float): angulo en grados. Solo se puede rotar en angulos rectos (0° 90° 180° 270° y 360°)
        """ 
        if angle % 90 == 0:
            self.angle = angle % 360

    def _rotate_image(self,image, angle):
        """Rotar imagen angulos rectos 

        Args:
            image (np.array): Imagen a rotar
            angle (int): angulo en grados (0°, 90°, 180°, 270°)

        Returns:
            np.array: imagen rotada
        """

        rotated = image.copy()

        if self.angle == 90:
            rotated  = cv2.flip(cv2.transpose(rotated), -1)
        if self.angle == 270:
            rotated  = cv2.flip(cv2.transpose(rotated), 1)
        if self.angle == 180:
            rotated = cv2.flip(rotated, -1)  # -1 indica voltear ambas dimensiones

        return rotated
    
    def disconnect(self):
        """Desconecta la camara 
        """

        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Cámara desconectada.")
        else:
            print("La cámara ya está desconectada.")
