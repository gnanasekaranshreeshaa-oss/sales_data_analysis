import pandas as pd
from prophet import Prophet
import streamlit as st

@st.cache
def load_data():
    df = pd.read_csv('train_egg_sales.csv', delimiter=';')
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Date': 'ds', 'Egg Sales': 'y'}, inplace=True)
    return df

def train_model(df):
    model = Prophet()
    model.fit(df)
    return model

def predict(model, future_dates):
    future = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def main():
    st.title("Egg Sales Forecasting")
    df = load_data()
    model = train_model(df)
    
    dates_to_predict = st.text_area("Enter dates to predict (comma separated, YYYY-MM-DD):")
    if st.button("Predict"):
        if dates_to_predict:
            dates_list = [d.strip() for d in dates_to_predict.split(',')]
            prediction = predict(model, dates_list)
            st.write(prediction)

if __name__ == "__main__":
    main()
