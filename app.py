import numpy as np
import pandas as pd
import streamlit as st
import pickle


#Load the saved model
f = open('trained_model_diabetes.sav', 'rb')
loaded_model = pickle.load(f)   
# loaded_model= pickle.load(open('trained_model_diabetes.sav','rb'))

#Create a function for prediction
def diabetes_prediction(input_data):
        #Changing the input data to numpy array
    inputdataasnumpyarray=np.asarray(input_data)

    #reshape thearry as we predict for one instancr
    inputreshape=inputdataasnumpyarray.reshape(1,-1)

    #standadize the input data
    # std_data=stdscalar.fit_transform(inputreshape)

    pred_res=loaded_model.predict(inputreshape)

    if pred_res[0]==0:
        return "The person is not diabetes"
    else:
        return "The person is diabetes"
    
def main():

    #Giving title
    st.title("Diabetes prediction app")

    # geting input data
    Pregnancies=st.text_input("Enter pregnancies value")
    Glucose=st.text_input("Enter Glucose value")
    BloodPressure=st.text_input("Enter BloodPressure value")
    SkinThickness=st.text_input("Enter SkinThickness value")
    Insulin=st.text_input("Enter Insulin value")
    BMI=st.text_input("Enter BMI value")
    DiabetesPedigreeFunction=st.text_input("Enter DiabetesPedigreeFunction value")
    Age=st.text_input("Enter Age value")

    #code for prediction
    Diagnosis=""

    #Create a button
    if st.button("Diabetes test result"):
        inputdata=(Pregnancies,Glucose,BloodPressure,SkinThickness,
                                      Insulin,BMI,DiabetesPedigreeFunction,Age)
        Diagnosis=diabetes_prediction(inputdata)
        
    st.success(Diagnosis)

if __name__=="__main__":
    main()