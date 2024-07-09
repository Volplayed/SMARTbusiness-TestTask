import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_data():
    df_customers = pd.read_csv('S_Data/customers.csv')
    df_geolocation = pd.read_csv('S_Data/geolocation.csv')
    df_order_items = pd.read_csv('S_Data/order_items.csv')
    df_order_payments = pd.read_csv('S_Data/order_payments.csv')
    df_order_reviews = pd.read_csv('S_Data/order_reviews.csv')
    df_orders = pd.read_csv('S_Data/orders.csv')
    df_products = pd.read_csv('S_Data/products.csv')
    df_sellers = pd.read_csv('S_Data/sellers.csv')
    df_product_category_name_translation = pd.read_csv('S_Data/product_category_name_translation.csv')

    #convert dates to datetime
    df_order_items['shipping_limit_date'] = pd.to_datetime(df_order_items['shipping_limit_date'])
    df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])
    df_orders['order_approved_at'] = pd.to_datetime(df_orders['order_approved_at'])
    df_orders['order_delivered_carrier_date'] = pd.to_datetime(df_orders['order_delivered_carrier_date'])
    df_orders['order_delivered_customer_date'] = pd.to_datetime(df_orders['order_delivered_customer_date'])
    df_orders['order_estimated_delivery_date'] = pd.to_datetime(df_orders['order_estimated_delivery_date'])

    #create merged dataframe
    data = pd.merge(df_orders, df_order_items, on='order_id')
    data = pd.merge(data, df_products, on='product_id')
    data = pd.merge(data, df_sellers, on='seller_id')
    data = pd.merge(data, df_customers, on='customer_id')
    data = pd.merge(data, df_order_payments, on='order_id')
    data = pd.merge(data, df_order_reviews, on='order_id')

    data['revenue'] = data['price'] * data['order_item_id']

    data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
    data['order_purchase_date'] = data['order_purchase_timestamp'].dt.date
    data['order_purchase_month'] = data['order_purchase_timestamp'].dt.to_period('M')

    data['order_purchase_day'] = data['order_purchase_timestamp'].dt.to_period('D')
    data['order_purchase_hour'] = data['order_purchase_timestamp'].dt.hour
    data['order_purchase_weekday'] = data['order_purchase_timestamp'].dt.dayofweek

    #drop columns that are not needed
    data = data.drop(['customer_zip_code_prefix', 'seller_zip_code_prefix', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'customer_unique_id', 'customer_zip_code_prefix', 'seller_zip_code_prefix', 'review_comment_title', 'review_comment_message', 'review_creation_date', 'review_answer_timestamp'], axis=1)


    return data

def train_best_classical_model(data):
    sales = data.groupby('order_purchase_day')["revenue"].sum()
    sales.index = sales.index.to_timestamp() 
    sales = sales.resample('D').sum().fillna(0)

    #fit model
    model = ARIMA(sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model.fit()

    return model

def train_best_ml_model(data):
    sales = data.groupby('order_purchase_day')["revenue"].sum()
    sales.index = sales.index.to_timestamp() 
    sales = sales.resample('D').sum().fillna(0)

    sales = sales.reset_index()
    sales['day'] = sales['order_purchase_day'].dt.day
    sales['month'] = sales['order_purchase_day'].dt.month
    sales['year'] = sales['order_purchase_day'].dt.year
    sales['weekday'] = sales['order_purchase_day'].dt.dayofweek
    sales['quarter'] = sales['order_purchase_day'].dt.quarter
    sales['dayofyear'] = sales['order_purchase_day'].dt.dayofyear
    sales['is_month_start'] = sales['order_purchase_day'].dt.is_month_start
    sales['is_month_end'] = sales['order_purchase_day'].dt.is_month_end
    sales['is_quarter_start'] = sales['order_purchase_day'].dt.is_quarter_start
    sales['is_quarter_end'] = sales['order_purchase_day'].dt.is_quarter_end
    sales['is_year_start'] = sales['order_purchase_day'].dt.is_year_start
    sales['is_year_end'] = sales['order_purchase_day'].dt.is_year_end

    X = sales.drop(['order_purchase_day', 'revenue'], axis=1)
    y = sales['revenue']

    model = RandomForestRegressor()
    model.fit(X, y)

    return model

def forecast_classical(days):
    sales = data.groupby('order_purchase_day')["revenue"].sum()
    sales.index = sales.index.to_timestamp() 
    sales = sales.resample('D').sum().fillna(0)
    future_dates = pd.date_range(start=sales.index.max(), periods=days, freq='D')
    future_dates = pd.DataFrame(future_dates, columns=['order_purchase_day'])
    

    preds = classical_model.fit().get_forecast(steps=days).predicted_mean

    future_dates['order_purchase_day'] = pd.to_datetime(future_dates['order_purchase_day']).dt.to_period('D')

    #plot
    plt.figure(figsize=(12,6))
    plt.plot(sales.index, sales, label='Actual')
    plt.plot(future_dates['order_purchase_day'], preds, label='Forecast')
    plt.title('Daily Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    return future_dates, preds

def forecast_ml(days):
    sales = data.groupby('order_purchase_day')["revenue"].sum()
    sales.index = sales.index.to_timestamp() 
    sales = sales.resample('D').sum().fillna(0)

    sales = sales.reset_index()

    future_dates = pd.date_range(start=sales["order_purchase_day"].max(), periods=14, freq='D')
    future_dates = pd.DataFrame(future_dates, columns=['order_purchase_day'])
    future_dates['day'] = future_dates['order_purchase_day'].dt.day
    future_dates['month'] = future_dates['order_purchase_day'].dt.month
    future_dates['year'] = future_dates['order_purchase_day'].dt.year
    future_dates['weekday'] = future_dates['order_purchase_day'].dt.dayofweek
    future_dates['quarter'] = future_dates['order_purchase_day'].dt.quarter
    future_dates['dayofyear'] = future_dates['order_purchase_day'].dt.dayofyear
    future_dates['is_month_start'] = future_dates['order_purchase_day'].dt.is_month_start
    future_dates['is_month_end'] = future_dates['order_purchase_day'].dt.is_month_end
    future_dates['is_quarter_start'] = future_dates['order_purchase_day'].dt.is_quarter_start
    future_dates['is_quarter_end'] = future_dates['order_purchase_day'].dt.is_quarter_end
    future_dates['is_year_start'] = future_dates['order_purchase_day'].dt.is_year_start
    future_dates['is_year_end'] = future_dates['order_purchase_day'].dt.is_year_end
    
    preds = ml_model.predict(future_dates.drop('order_purchase_day', axis=1))

    #plot
    plt.figure(figsize=(12,6))
    plt.plot(sales['order_purchase_day'], sales['revenue'], label='Actual')
    plt.plot(future_dates['order_purchase_day'], preds, label='Forecast')
    plt.title('Daily Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    return future_dates, preds


if __name__ == "__main__":
    data = get_data()
    classical_model = train_best_classical_model(data)
    ml_model = train_best_ml_model(data)
    days = int(input("Enter number of days to forecast: "))
    model = input("Enter model to use (classical or ml): ")
    if model == 'classical':
        forecast_classical(days)
    elif model == 'ml':
        forecast_ml(days)
    else:
        print("Invalid model entered")
