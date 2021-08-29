import streamlit as st
import tensorflow as tf
from PIL import Image
from utilities.preprocessing import preprocessing_image
from utilities.gradcam import VizGradCAM
import numpy as np

import warnings
warnings.filterwarnings("ignore")

@st.cache
def load_model():
    
    model=tf.keras.models.load_model("./model/best-model-deeplearning-detector-COVID19.h5",compile=False)
    return model


def load_image(image_bytes,column=None):
    img=Image.open(image_bytes)
    column.image(img.resize((300,300)),caption="Radiografia X-Ray")
    array_img=preprocessing_image(image_bytes,expand_dims=True)
    return array_img

@st.cache
def get_prediction(model,image_array):

    prediction=model.predict(image_array)
    
    return prediction[0]

def show_image_gradcam(model,image_array,col):
    grad_cam=VizGradCAM(model,image_array,interpolant=0.36)
    img_gc=Image.fromarray(grad_cam)
    col.image(img_gc.resize((300,300)),caption="GRAD-CAM")

def show_results(predictions):
    class_result=np.argmax(predictions)
    if class_result==0:
        st.markdown("La radiografia corresponde a un paciente con **COVID-19**. Por su salud, acuda a un centro de vacunacion")
    elif class_result==1:
        st.markdown("La radiografia corresponde a un paciente con una **ENFERMEDAD PULMONAR** distinta al COVID-19. Acuda a un medico para su tratamiento")
    elif class_result==2:
        st.markdown("La radiografia  de rayos-X corresponde a un **paciente NORMAL**")
    st.subheader("Precision de diagnostico")
    col1,col2,col3=st.columns(3)
    col1.metric("COVID-19","{0:0.2f}%".format(predictions[0]*100))
    col2.metric("Enfermedades pulmonares","{0:0.2f}%".format(predictions[1]*100))
    col3.metric("Pulmones Normales","{0:0.2f}%".format(predictions[2]*100))

def main():
    st.title("Modelo para la deteccion de COVID-19 en base a radiografias")
    file_image=st.file_uploader("Seleccione un archivo de imagen",type=["png","jpg","jpeg"])

    model=load_model()
    predictions=None
   
    if file_image is not None:
        col1,col2=st.columns(2)
        image_array=load_image(file_image,col1)
        
        array_img=preprocessing_image(file_image,expand_dims=False)
        show_image_gradcam(model,array_img,col2)
        predictions=get_prediction(model,image_array)
        show_results(predictions)


if __name__=="__main__":
    main()