import streamlit as st
from PIL import Image
from utilities.preprocessing import datauri2pil, pil2datauri
import numpy as np
import requests

import warnings
warnings.filterwarnings("ignore")

im = Image.open("resources/redv2.jpg")
st.set_page_config(
        page_title="Deteccion de COVID19",
        page_icon=im,
       
)


def show_image_xray(image_bytes,column=None):
    img=Image.open(image_bytes)
    column.image(img.resize((300,300)),caption="Radiografia X-Ray")



def show_image_gradcam(img_gc,column):
    column.image(img_gc.resize((300,300)),caption="GRAD-CAM")

def consume_api_detector(content):
    API_URL="https://apicovid19detector-5ou3s2kqgq-uc.a.run.app/predict"
    data={
        "image":content
    }
    response=requests.post(API_URL,json=data)
    if response.status_code==200:
        return response.json()
    else:
        return None

def show_results(predictions):
    class_result=np.argmax(predictions)
    if class_result==0:
       
        st.error("La radiografia corresponde a un paciente con **COVID-19**. Por su salud, acuda a un centro de vacunacion mas Cercano.")
    elif class_result==1:
        st.warning("La radiografia corresponde a un paciente con una **ENFERMEDAD PULMONAR** distinta al COVID-19. Acuda a un medico para su tratamiento. Posiblemente se trate de neumonia viral o Lung Opacity.")
    elif class_result==2:
        st.success("La radiografia  de rayos-X corresponde a un **PACIENTE NORMAL**")
    st.subheader("Precision de diagnostico")
    col1,col2,col3=st.columns(3)
    col1.metric("COVID-19","{0:0.2f}%".format(predictions[0]*100))
    col2.metric("Enfermedades pulmonares","{0:0.2f}%".format(predictions[1]*100))
    col3.metric("Pulmones Normales","{0:0.2f}%".format(predictions[2]*100))

def main():
    st.title("Modelo para la deteccion de COVID-19 en base a radiografias")
    st.subheader("Desarrollador: Johan Mitma - UNMSM -2021")
    file_image=st.file_uploader("Seleccione un archivo de imagen",type=["png","jpg","jpeg"])

    #model=load_model()
    predictions=None
   
    if file_image is not None:
        col1,col2=st.columns(2)
        show_image_xray(file_image,col1)
        prediction=consume_api_detector(pil2datauri(Image.open(file_image)))

        if prediction is not None:
            show_image_gradcam(datauri2pil(prediction["gradcam"]),col2)
            predictions=prediction["prediction"]
            show_results(predictions[0])
        else:
            st.error("La API no se encuentra disponible")
    else:
        st.success("Por favor, suba una radiografia de rayos-X para su descarte de COVID-19")
        
        st.image(Image.open("./resources/portada_xray.jpg"))
    st.markdown("*Investigacion dedicada a mi madre **Isabel Huaccha Fernandez** *")
    st.markdown("**Usar responsablemente:** Modelo con **92%** de precision")
    st.markdown("**Acerca del autor**: [Linkedin](https://www.linkedin.com/in/johan-valerio-mitma12/) | [IG](https://www.instagram.com/johan_mitma12/)")

if __name__=="__main__":
    main()