# импорт библиотек для построения dagов
import pendulum
from airflow.decorators import dag, task
from datetime import timedelta

@dag(
    schedule = timedelta(days = 10),
    start_date = pendulum.datetime(2024, 9, 30, hour = 13, tz = "UTC"),
    dag_id = 'linear_reg',
    tags = ['po_prices']
)

def linearReg_predictions():
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    @task()
    def extract_dataset():
        raw_data = pd.read_excel("~\Desktop\po-prices_pipeline\datasets\dataset_po modeling.xlsx")
        return raw_data
    
    @task()
    def transform(data: pd.DataFrame):
        data.columns = data.columns.str.lower()

        # найдем дату последнего пустого значения по столбцу mean_temp
        last_isna_date = data[data['mean_temp'].isnull() == True]['date'].to_list()[-1]
        data = data[data['date'] > last_isna_date]

        def shifting_values(series, number):
            data[series + "_" + str(number)] = data[series].shift(number)

        # создание новых столбцов со смещенными значениями
        for i in [1, 2, 4]:
            shifting_values('palm oil_rbd', i)
    
        # создадим столбцы со смещенными значениями для остальных параметров 
        shifting_values('rupees/$', 2)
        shifting_values('yuan/$', 9)
        shifting_values('euro/$', 8)
        shifting_values('ruble/$', 3)
        shifting_values('ringgit/$', 2)
        shifting_values('oil_brent', 6)
        shifting_values('oil_urals', 6)
        shifting_values('mean_temp', 1)
        shifting_values('mean_pres', 2)
        shifting_values('mean_humid', 1)

        data = data[data['date'] > data[data['yuan/$_9'].isna() == True]['date'].to_list()[-1]]

    @task()
    def evaluate_model(data: pd.DataFrame):
        new_columns_list = []
        new_columns_list.extend(['date', 'palm oil_rbd'])

        for i in data.columns.to_list():
            try:
                if int(i[-1]) in range(1, 11):
                    new_columns_list.append(i)
            except:
                continue

        data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))
        df = data[new_columns_list].set_index('date')

        date_series = pd.Series(df.index)
        filter_date = date_series.quantile(0.8)

        train_data = df[df.index < filter_date]
        test_data = df[df.index >= filter_date]

        new_columns_list.remove('date')
        new_columns_list.remove('palm oil_rbd')

        X_train = train_data[new_columns_list]
        X_test = test_data[new_columns_list]
        y_train = train_data['palm oil_rbd']
        y_test = test_data['palm oil_rbd']

        mm = MinMaxScaler()

        X_train_std = mm.fit_transform(X_train)
        X_test_std = mm.transform(X_test)

        reg = LinearRegression()
        reg.fit(X_train_std, y_train)

        data = data.set_index('date')
        data = pd.DataFrame(data = mm.fit_transform(data), columns = data.columns)

        columns_list = data.columns.to_list()

        # список, где будут храниться значения для дальнейшего прогноза
        values_list = []
        for i in columns_list:
            try:
                number = int(i[-1])
                feature = i[:-2]
                values_list.append(df[feature].values[-number])
            except:
                continue

        values_array = np.array(values_list)

        pred = reg.predict(values_array.reshape(1, -1))
        pred = pd.DataFrame({'sklearn_values': pred,
                               'date': max(data.index.date) + timedelta(days = 1)})
        pred = pred.set_index('date')

        return pred
    
    @task()
    def load(pred: pd.DataFrame):
        pred.to_csv("C:\Users\k.storozhuk\Desktop\po-prices_pipeline\predictions")

    data = extract_dataset()
    transformed_data = transform(data)
    predictions= evaluate_model(transformed_data)
    load(predictions)

linearReg_predictions()