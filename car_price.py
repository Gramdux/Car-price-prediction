import pandas as pd
import numpy as np
import pypickle as pk
import pickle
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import streamlit as st



loaded_model = pk.load('model.pkl')

def prediction(data):

     df = pd.DataFrame(data)
     df.iloc[6].replace({"Manual": 1, 'Automatic': 0}, inplace=True)
     df.iloc[4].replace({'Petrol': 1, 'Diesel': 0, 'CNG': 2}, inplace=True)
     df.iloc[5].replace({'Dealer': 1,'Individual': 0}, inplace=True)



     label = preprocessing.LabelEncoder()

     df.iloc[0] = label.fit_transform(df.iloc[0])

     num_data = df.drop([7]).values.reshape(1, -1)

     pred = loaded_model.predict(num_data)

     
     return f"The Price of this car is {pred}"



def main():
     
     st.title("Car price prediction")
     Car_Name = st.text_input("What is the name of the car")
     Year = st.number_input("What is the year of the car")
     Present_Price = st.number_input("What is the present price of the car")
     Driven_kms = st.number_input("What is the driven_kms of the car")
     Fuel_Type = st.text_input("What is the fuel_type of the car")
     Selling_type = st.text_input("What is the selling_type of the car")
     Transmission = st.text_input("What is the transmission of the car")
     Owner = st.number_input("Who is the owner of the car")
    
     selling_price = " "


     if st.button("Result"):
          selling_price = prediction([Car_Name, Year, Present_Price, Driven_kms,
       Fuel_Type, Selling_type, Transmission, Owner])
          

     st.success(selling_price)



if __name__ == "__main__":
     main()
     
