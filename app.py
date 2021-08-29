import streamlit as st
import tensorflow as tf
from PIL import Image
from utilities.preprocessing import preprocessing_image
from utilities.gradcam import VizGradCAM


@st.cache
def load_model():
    st.write("Loading model...")
    
    model=tf.keras.models.load_model("./model/best-model-deeplearning-detector-COVID19.h5",compile=False)
    st.write("Model loaded.")
    return model


def load_image(image_bytes):
    img=Image.open(image_bytes)
    st.image(img.resize((256,256)),caption="Radiografia")
    array_img=preprocessing_image(image_bytes,expand_dims=True)
    return array_img

@st.cache
def get_prediction(model,image_array):

    prediction=model.predict(image_array)
    
    return prediction[0]

def show_results(predictions):
    st.subheader("Diagnostico")
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
        image_array=load_image(file_image)
        predictions=get_prediction(model,image_array)
        show_results(predictions)


if __name__=="__main__":
    main()