import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


st.title('Egg Sales Analysis & Forecasting')
train_file = st.file_uploader("Upload train data", type="csv")
test_file = st.file_uploader("Upload test data", type="csv")

if train_file and test_file:
    train_df = pd.read_csv(train_file, sep=';')
    test_df = pd.read_csv(test_file)
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    
    st.subheader('Historical Sales Trend')
    fig, ax = plt.subplots()
    ax.plot(train_df['Date'], train_df['Egg Sales'], label='Egg Sales')
    st.pyplot(fig)

   
    st.subheader('Forecast Sales')
    prophet_df = train_df.rename(columns={'Date': 'ds', 'Egg Sales': 'y'})
    model = Prophet()
    model.fit(prophet_df)

    future = test_df.rename(columns={'Date': 'ds'})
    forecast = model.predict(future)
    test_df['Egg Sales Prediction'] = forecast['yhat']
    st.write(test_df)

   
    st.download_button(
        label="Download Predictions",
        data=test_df.to_csv(index=False),
        file_name="egg_sales_predictions.csv",
        mime="text/csv"
    )