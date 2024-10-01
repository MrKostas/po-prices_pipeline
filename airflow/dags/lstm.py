# импорт библиотек для построения dagов
import pendulum
from airflow.decorators import dag, task
from datetime import timedelta

@dag(
    schedule = timedelta(days = 10),
    start_date = pendulum.datetime(2024, 9, 30, hour = 13, tz = "UTC"),
    dag_id = 'lstm_model',
    tags = ['po_prices']
)

def lstm_predictions():
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM, Dropout
    from keras.optimizers import Adam

    @task()
    def extract_dataset():
        raw_data = pd.read_excel("~\Desktop\po-prices_pipeline\datasets\dataset_po modeling.xlsx")
        data = raw_data[['date', 'Palm oil_RBD']]
        return data
    
    @task()
    def transform(data: pd.DataFrame):
        data.columns = data.columns.str.lower()
        data['date'] = data['date'].apply(lambda x: pd.to_datetime(x))
        data['palm oil_rbd'] = data['palm oil_rbd'].astype('float')

        data = data.sort_values(by = 'date', ascending = True)
        data = data[['date', 'palm oil_rbd']]

        # поиск даты последнего пустого значения из столбца Palm Oil_RBD
        border_date = data[data['palm oil_rbd'].isna() == True]['date'].to_list()[-1]
        data = data[data['date'] > border_date]
        data = data.set_index('date')

        # устанавливается граница раздела обучающей и тестовой выборки - 95% от общего набора отводится на обучение нейронной сети
        num_shape = int(data.shape[0] * 0.95)

        # делим общий набор на обучающий и тестовый
        train = data.iloc[:num_shape, :].values
        test = data.iloc[num_shape:, :].values

        sc = MinMaxScaler(feature_range=(0, 1))
        train_sc = sc.fit_transform(train)

        X_train = [] # список для хранения каждого из окон
        y_train = [] # каждое последующее значение после окна - то, которое необходимо спрогнозировать

        window = 50 # формируем окна из 50 значений

        for i in range(window, num_shape):
            X_train_ = np.reshape(train_sc[i - window : i, 0], (window, 1))
            X_train.append(X_train_)
            y_train.append(train_sc[i, 0])

        X_train = np.stack(X_train)
        y_train = np.stack(y_train)

        df_volume = np.vstack((train, test))
        inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        
        num_2 = df_volume.shape[0] - num_shape + window

        X_test = []

        for i in range(window, num_2):
            X_test_ = np.reshape(inputs[i - window : i, 0], (window, 1))
            X_test.append(X_test_)

        X_test = np.stack(X_test)

        return X_train, y_train, X_test

    @task()
    def evaluate_model(X_train: np.array, y_train: np.array, X_test: np.array, num_2: int, data: pd.DataFrame):
        num_shape = int(data.shape[0] * 0.95)
        train = data.iloc[:num_shape, :].values
        sc = MinMaxScaler(feature_range=(0, 1))
        train_sc = sc.fit_transform(train)

        # инициализация объекта реккурентной нейронной сети
        modelLSTM = Sequential()
        modelLSTM.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 1)))
        modelLSTM.add(Dropout(0.2))
        modelLSTM.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
        modelLSTM.add(Dropout(0.2))
        modelLSTM.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
        modelLSTM.add(Dropout(0.2))
        modelLSTM.add(LSTM(units = 120, activation = 'relu'))
        modelLSTM.add(Dropout(0.2))
        modelLSTM.add(Dense(units = 1))
        
        modelLSTM.compile(optimizer = 'adam', loss = 'mean_squared_error')
        modelLSTM.fit(X_train, y_train, epochs = 400, batch_size = 32)

        predict = modelLSTM.predict(X_test)
        predict = sc.inverse_transform(predict)

        # прогноз на 10 дней вперед

        # задаем изначальные параметры для модели
        pred_ = predict[-1].copy()
        prediction_full = []
        window = 50
        df_copy = data.values

        # цикл обучения и добавления в набор новых прогнозных значения
        for j in range(10):
            df_ = np.vstack((df_copy, pred_))
            train_ = df_[:num_shape]
            test_ = df_[num_shape:]
            df_volume_ = np.vstack((train_, test_))
            inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
            inputs_ = inputs_.reshape(-1, 1)
            inputs_ = sc.transform(inputs_)
            X_test_2 = []
            for k in range(window, num_2):
                X_test_3 = np.reshape(inputs_[k - window : k, 0], (window, 1))
                X_test_2.append(X_test_3)
                
                X_test_ = np.stack(X_test_2)
                predict_ = modelLSTM.predict(X_test_)
                pred_ = sc.inverse_transform(predict_)
                prediction_full.append(pred_[-1][0])
                df_copy = df_[j:]

        date_range = pd.date_range(start = pd.to_datetime(max(data.index.date) + pd.DateOffset(days = 1)),
                                       end = pd.to_datetime(max(data.index.date) + pd.DateOffset(days = 10)))
            
        pred = pd.DataFrame(data = np.array(prediction_full), index = date_range).rename(columns = {'0' : 'predicted_price'})

        return pred
        
    @task()
    def load(pred: pd.DataFrame):
        pred.to_csv("C:\Users\k.storozhuk\Desktop\po-prices_pipeline\predictions")

    
    data = extract_dataset()
    transformed_data = transform(data)
    predictions = evaluate_model(transformed_data[0], transformed_data[1], transformed_data[2], data)
    load(predictions)

lstm_predictions()