import tensorflow as tf
import io,base64
from PIL import Image,ImageOps
import numpy as np
def preprocessing_image(img_base,expand_dims=False,equalize=True):
    """
    array_bytes: Imagen extraida de consulta POST de una aplicacion que consume dicha API
    for_prediction: Indica si la imagen es para predicciones y su es TRUE se devuelve con una dimension de (1,...)

    return: Imagen procesada en una matriz de 4 dimensiones (n_samples,height,width,n_channels)
    """

    #array_bytes=base64.b64decode(img_base64)
    #image=Image.open(io.BytesIO(array_bytes))
    image=Image.open(img_base)
    image=image.resize((256,256),Image.ANTIALIAS) #antialias retonar la imagen con mejor calidad, pero cambiando el tamaño
    #determinamos si la imagen no esta en 3 canales de color
    if image.mode!="RGB":
        image=image.convert(mode="RGB") #la convertimos a RGB
    #ecualizamos la imagen
    if equalize:
        image=ImageOps.equalize(image,mask=None)
    #convertimos la imagen a una matriz de pixeles
    image_array=np.asarray(image)
    image_array=image_array.astype(np.float64)/255.0 #la normalizamos
    #convertimos la matriz de pixeles a una matriz de dimensiones (n_samples,height,width,n_channels)
    if expand_dims:
        image_array=np.expand_dims(image_array,axis=0)
    
    return  image_array

def pil2datauri(img):
    #converts PIL image to datauri
    data = io.BytesIO()
    img.save(data, "JPEG")
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

def datauri2pil(uri):
    #converts datauri to PIL image
    data = base64.b64decode(uri.split(',')[1])
    return Image.open(io.BytesIO(data))
