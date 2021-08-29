import numpy as np
import tensorflow.keras as K
from skimage.transform import resize
from tensorflow.keras.models import Model
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


def get_transfer_model(model):
    """
    Model: Modelo de machine learning con transferencia de aprendizaje

    return: Modelo principal de  Transfer Learning
    """
    model_transfer=None
    for layer in model.layers:
        if isinstance(layer,tf.keras.Model):
            model_transfer=layer
            break
    return model_transfer

def get_last_conv_layer(model):
    """
    Model: Modelo principal de transfer learning.

    return: La capa de convolucion mas profunda
    """
    for layer in model.layers[::-1]:
        if isinstance(layer,K.layers.Conv2D):
            return layer
    return None #return None si no encontramos la capa de convolucion

def VizGradCAM(model, image, interpolant=0.6, return_gradcam=True):
   
    """
    Funcion de GRADCAM

    model: Modelo de machine learning
    image: Imagen de entrada de 3 dimensiones.
    interpolant: Valor de interpolante de la imagen de entrada.

    return_gradcam: Si es True Indica si se retorna la imagen de salida de la gradcam, y si es
                     False, retorna la imagen del mapa de calor donde el modelo tuvo mayor actividad.
    """
    
    transfer_model=get_transfer_model(model)
   
    last_conv_layer = get_last_conv_layer(transfer_model)

    target_layer = transfer_model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)

    # Obtain Prediction Index
    prediction_idx = np.argmax(prediction)
    
    # Compute Gradient of Top Predicted Class
    with tf.GradientTape() as tape:
        gradient_model = Model([transfer_model.inputs], [target_layer.output, transfer_model.output])
        conv2d_out, prediction = gradient_model(img)
        # Obtain the Prediction Loss
        loss = prediction[:, prediction_idx]

    # Gradient() computes the gradient using operations recorded
    # in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    # Obtain the Output from Shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]
     # Obtain Depthwise Mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    # Create a 7x7 Map for Aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

    # Multiply Weights with Every Layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    # Resize to Size of Image
    activation_map = cv2.resize(
        activation_map.numpy(), (original_img.shape[1], original_img.shape[0])
    )

    # Ensure No Negative Numbers
    activation_map = np.maximum(activation_map, 0)

    # Convert Class Activation Map to 0 - 255
    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min()
    )
    activation_map = np.uint8(255 * activation_map)

    # Convert to Heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    # Superimpose Heatmap on Image Data
    original_img = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min())
        * 255
    )

    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Enlarge Plot
    plt.rcParams["figure.dpi"] = 100
    if return_gradcam == True:
        return np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))
        
        #plt.savefig("./grad_cam_image.png")
    else:
        return cvt_heatmap