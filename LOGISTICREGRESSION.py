import pickle
import streamlit as st
import joblib

with open("LOGISTICREGRESSION.pkl", "rb") as pickle_in:
    regressor = pickle.load(pickle_in)


def predict_survived(gender, age, sibsp, parch, fare):
    prediction = regressor.predict([[gender, age, sibsp, parch, fare]])
    return prediction

def main():
    st.title("Прогноз выживания на Титанике")
    st.markdown("Учебная модель поможет предсказать выживет ли человек на Титанике. Модель обучена на основе данных Titanic с использованием алгоритма GrdBosstingReg (точность 76%).")
    
    gender = st.radio("Пол:", options=["Мужской", "Женский"])
    age = st.slider("Возраст:", min_value=0, max_value=100, step=1)
    sibsp = st.slider("Количество братьев/сестер на борту:", min_value=0, max_value=10, step=1)
    parch = st.slider("Количество родителей/детей на борту:", min_value=0, max_value=10, step=1)
    fare = st.number_input("Стоимость билета:", min_value=0.0, format="%.2f")
    
    if st.button("Предсказать"):
        prediction = predict_survived(gender, age, sibsp, parch, fare)
        st.success(f"Предсказание: {prediction}")

if __name__ == '__main__':
    main()
     

