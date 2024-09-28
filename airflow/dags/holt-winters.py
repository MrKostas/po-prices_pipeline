# импорт библиотек для построения dagов
import pendulum
from airflow.decorators import dag, task
from datetime import timedelta

@dag(
    schedule = timedelta(days = 10),
    start_date = pendulum.datetime(2024, 9, 30, hour = 13, tz = "UTC"),
    dag_id = 'holt-winters',
    tags = ['po_prices']
)

def holtw_predictions():
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    @task()
    def extract_dataset():
        raw_data = pd.read_excel("~\Desktop\po-prices_pipeline\datasets\dataset_po modeling.xlsx")
        data = raw_data[['date', 'Palm oil_RBD']]
        return data
    
    @task()
    def transform(data: pd.DataFrame):
        # приводим заголовки столбцов к snake_case
        data.columns = data.columns.str.lower()

        # контрольная обработка столбцов с датой и ценой пальмового масла
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))
        data['palm oil_rbd'] = data['palm oil_rbd'].astype('float')

        #сортировка датасета по возрастанию даты
        data = data.sort_values(by = 'date', ascending = True)

        # поиск даты последнего пустого значения из столбца Palm Oil_RBD
        border_date = data[data['palm oil_rbd'].isna() == True]['date'].to_list()[-1]
        
        # отфильтровываем набор данных по дате
        data = data[data['date'] > border_date]

        return data

    @task()
    def evaluate_model(data: pd.DataFrame):
        # обозначение границ прогноза
        start = len(data.index)
        end = len(data['po_price']) + 29

        # обучаем модель экспоненциального сглаживания
        model = ExponentialSmoothing(endog = data,
                                          trend = 'mul',
                                          seasonal = 'mul',
                                          seasonal_periods = 365).fit(optimized = True)
        
        # формируем прогноз в формате DataFrame
        pred = pd.DataFrame(data = round(model.predict(start, end), 2),
                            index = pd.date_range(start = pd.to_datetime(max(data.index.date) + timedelta(days = 1),
                                                                         format = '%Y-%m-%d'), freq = 'D', periods = 30))
        
        # задаем названия столбцу индекса и столбцу значений
        pred.index.name = "date"
        pred = pred.rename(columns = {0: 'holtw_values'})

        return pred
    
    @task()
    def load(pred: pd.DataFrame):
        pred.to_csv("C:\Users\k.storozhuk\Desktop\po-prices_pipeline\predictions")

    
    data = extract_dataset()
    transformed_data = transform(data)
    predictions = evaluate_model(transformed_data)
    load(predictions)

holtw_predictions()