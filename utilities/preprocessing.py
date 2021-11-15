import io,base64
from PIL import Image


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
