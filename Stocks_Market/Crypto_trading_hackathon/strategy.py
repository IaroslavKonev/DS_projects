import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
from typing import Optional


class Strategy(ABC):

    @abstractmethod
    def required_rows(self):
        raise NotImplementedError("Specify required_rows!")

    @abstractmethod
    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        assert len(current_data) == self.required_rows  # This much data will be fed to model

        return None  # If None is returned, no action is executed


class YourStrategy(Strategy):
    required_rows = 2 * 24 * 60  # Specify how many minutes of data are required for live prediction

    def __init__(self):
        '''
        When the class is initialized, the training dataset is loaded and the model is trained
        '''
        training_data = pd.read_csv("data/train_data.csv", parse_dates=["time"]).set_index('time')
        self.encoder = self.custom_one_hot_encoder(self.make_features(training_data, need_lag=False))
        self.model = self.train_model(training_data)
        self.TOD = 0
        self.predicted_price = [0]


    def make_features(self, data, need_lag=True, max_lag=1):
        '''
        :param need_lag: bool, if you want to add lags use True
        :param max_lag: int, number of lags for timeseries
        :param data: DataFrameObject, Dataset for training model or for prediction

        :return: data, DataFrameObject, Dataset with new features for training model or for prediction

        This function do feature engineering job. Hours, days of the week and months are retrieved from the index.
        Also make n-m price-lag.
        '''
        data['hours'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['month'] = data.index.month

        if need_lag:
            for lag in range(1, max_lag + 1):
                data['lag_{}'.format(lag)] = data['price'].shift(lag)

            return data

        else:
            return data

    def custom_one_hot_encoder(self, data):
        '''
        :param data: DataFrameObject, Dataset for training model or for prediction

        :return: enc, trained OHE-model

        This function train and return OHE-model for new categorical features
        '''
        enc = OneHotEncoder()
        enc.fit(data[['month', 'dayofweek', 'hours']])

        return enc

    def windowed_df_to_date_X_y(self, windowed_dataframe, train=True):
        '''
        :param windowed_dataframe: DataFrameObject - Frame with all features

        :return X : np.array, array with features
                Y : np.array, array with targets

        This function separate features and targets
        '''
        if train:
            y = windowed_dataframe['price']
            X = windowed_dataframe.drop(['price', 'time'], axis=1)

            return X, y

        else:
            X = windowed_dataframe.drop(['datetime'], axis=1)

            return X

    def pipeline(self, data, train=False):
        '''
        :param data: DataFrameObject, Dataset for training model or for prediction
        :param train: bool, this param choose the

        :return: if train=False - X, DataFrameObject with features for prediction
                 if train=True - X, DataFrameObject - with features for training
                                y, Series with target for training

        This option selects the processing branch of the frame to prepare the data
        for training or prediction.
        '''
        data = data.reset_index()
        if train:

            data = data[["time", "price"]]
            data = data[data['time'] > '2020-07-01']
            data = data.set_index("time")

            # Do resampling
            data = data.resample('5Min').mean()

            data = self.make_features(data, max_lag=3)

            df_ohe = self.encoder.transform(data[['month', 'dayofweek', 'hours']])
            df_ohe = pd.DataFrame(df_ohe.toarray())
            df_ohe = df_ohe.astype('int')
            data = data.reset_index()
            data = data.join(df_ohe, how="left")

            data = data[['time', 'lag_1', 'lag_2', 'lag_3', 0, 1,
                         2, 3, 4, 5, 6,
                         7, 8, 9, 10, 11,
                         12, 13, 14, 15, 16,
                         17, 18, 19, 'price']]

            data = data.dropna()

            # apply function
            X, y = self.windowed_df_to_date_X_y(data, train=train)

            return X, y

        else:
            data = data[["datetime", "price"]]
            data = data.set_index("datetime")

            # do resampling
            data = data.resample('5Min').mean()

            data = self.make_features(data, max_lag=2)

            df_ohe = self.encoder.transform(data[['month', 'dayofweek', 'hours']])
            df_ohe = pd.DataFrame(df_ohe.toarray())
            df_ohe = df_ohe.astype('int')
            data = data.reset_index()
            data = data.join(df_ohe, how="left")

            data = data[['datetime', 'price', 'lag_1', 'lag_2', 0, 1,
                         2, 3, 4, 5, 6,
                         7, 8, 9, 10, 11,
                         12, 13, 14, 15, 16,
                         17, 18, 19]]

            data = data.dropna()

            # apply function
            X = self.windowed_df_to_date_X_y(data, train=train)

            return X

    def root_mse(self, y, y_pred):
        '''
        :param y: DataSeries, target for test
        :param y_pred: , list, with predictions

        :return: RMSE

        This function compute RMSE-metric
        '''
        return (mean_squared_error(y, y_pred)) ** 0.5

    def gen_model(self, X, y):
        '''
        :param X: DataFrameObject, Frame with features
        :param y: DataSeries, There are targets

        :return: trained model

        This function trains LR-model. Grid search is used for optimization
        '''

        rmse = make_scorer(self.root_mse, greater_is_better=False)
        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}

        scv = TimeSeriesSplit(n_splits=10)

        LR_model_GSCV = GridSearchCV(LinearRegression(),
                                     param_grid=param_grid,
                                     scoring=rmse,
                                     cv=scv,
                                     verbose=True)

        best_LR = LR_model_GSCV.fit(X, y)
        model = LinearRegression(**best_LR.best_params_)
        model.fit(X, y)

        return model

    def train_model(self, training_data):
        '''
        :param training_data: DataFrameObject, Initial data frame for preprocessing

        :return: trained model
        '''

        return self.gen_model(*self.pipeline(training_data, train=True))


    def compute_target_position(self, current_data: pd.DataFrame, current_position: float) -> Optional[float]:
        '''
        :param current_data:
        :param current_position:

        :return: target_position

        This function used one of the best algorithm for trading - MeanReverse, if we
        checked the chart, we would see that price always cross the MovingAverage finally.
        But in the end we are using mean between prediction price for next 5min
        and average price.
        '''
        k = current_data.resample('5Min').mean().index.minute[-1]

        if k==self.TOD:

            if self.predicted_price[-1]==0:

                try:
                    X = self.pipeline(current_data.tail(16), train=False)
                    self.predicted_price = self.model.predict(X.tail(1))

                except: self.predicted_price=[0]

            avg_price = current_data['price'].mean()
            current_price = current_data['price'][-1]

            target_position = current_position + (self.predicted_price[-1] - current_price) / 1500

            return target_position

        else:
            self.TOD=k
            X = self.pipeline(current_data.tail(16), train=False)
            try:
                self.predicted_price = self.model.predict(X.tail(1))
            except:
                self.predicted_price = [0]
            avg_price = current_data['price'].mean()
            current_price = current_data['price'][-1]

            target_position = current_position + (self.predicted_price[-1] - current_price) / 1500

            return target_position








































