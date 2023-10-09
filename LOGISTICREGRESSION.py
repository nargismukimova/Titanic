import numpy as np
import pickle
import pandas as pd 
import streamlit as st

from PIL import Image


pickle_in = open("LOGISTICREGRESSION.pkl","rb")
regressor=pickle.load(pickle_in)

def predict_survived(gender, Age, SibSp, Parch, Fare):
    
   
    prediction=regressor.predict([[gender, Age, SibSp, Parch, Fare]])
    print(prediction)
    return prediction



def main():
    st.title("Прогноз выжившив на Титанике")
    html_temp = """
    <div style="background-color:green    ;padding:10px">
    <h2 style="color:white;text-align:center;">Учебная модель поможет предсказать выживет ли человек на Титанике. МОдель обучена на основе данных titanic, алгоритмом GrdBosstingReg (точность 76%) </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    m=st.radio("Пол:",
        key="gender",
        options=["Мужской", "Женский"],    )
    if m=="Мужской":
        market_code=0
    elif m=="Женский":
        market_code=1
        
    
  #  rooms = st.text_input("Количество комнат","")
   # floor = st.text_input("Этаж","")
    #area = st.text_input("Площадь в кв.м.","")
    
   # r = st.radio("Ремонт:",
    #             key="Remodel",
     #            options=["Нет ремонта(коробка)", "Старый ремонт", "Новый ремонт"], )
   # if r == "Нет ремонта(коробка)":
    #    remodel_code = 0
    #elif r == "Старый ремонт":
     #   remodel_code = 1
    #elif r == "Новый ремонт":
     #   remodel_code = 2
    
    
    
    
#    result=""
 #   if st.button("Predict"):
  #      result=int(predict_house_price(market_code, rooms, floor, area, remodel_code))
    
   # st.success('Оценочная стоимость квартиры {}'.format(result)+" Сомони")
     
     
    
    
    

if __name__=='__main__':
    main()
    
