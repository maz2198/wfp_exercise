import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,TimeSeriesSplit
from prophet import Prophet# Visualization
import matplotlib.pyplot as plt
# Prophet model for time series forecast
from prophet import Prophet# Visualization
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

def get_season(mydate):
    """
    If the date is not null, try to get the day of the year, and if it's between 80 and 172, return 1,
    if it's between 172 and 264, return 2, if it's between 264 and 355, return 3, otherwise return 4
    
    :param mydate: The date you want to extract the season from
    :return: The season number (1,2,3,4)
    """
    if mydate is not None:
        try:
            season = 0
            dayofyear = mydate.timetuple().tm_yday
            spring = range(80,172)
            summer = range(172,264)
            autumn = range(264,355)
            if dayofyear in spring:
                season = 1
            elif dayofyear in summer:
                season = 2
            elif dayofyear in autumn:
                season = 3
            else:
                season = 4
            return season
        except Exception as e:
            print(f"Exception on extracting season with error:{e}")         

def generate_date_features(df):
    """
    It takes a dataframe with a datetime index and adds a bunch of new columns to it
    
    :param df: The dataframe to generate the features for
    :return: the dataframe with the new columns added.
    """
    df["month_name"] = df['date_decade'].dt.month_name()
    df["isstartofmonth"] = df['date_decade'].dt.is_month_start.astype("float")
    df["day_name"] = df['date_decade'].dt.day_name()
    df["dayofweek"] = df['date_decade'].dt.weekday
    df["isweekend"] = np.where(df['date_decade'].dt.dayofweek > 4, True, False).astype(float)
    df["weekofmonth"] = (df['date_decade'].dt.day / 7).astype(int) + 1
    df["quarter"] = df['date_decade'].dt.quarter
    df["isstartofquarter"] = df['date_decade'].dt.is_quarter_start.astype(float)
    df["season"] = df['date_decade'].apply(get_season)

    return df

def print_min_max_dates(df,date_col: str,type_date: str):
    """
    This function takes in a dataframe, a date column, and a type of date (e.g. "start" or "end") and
    prints the earliest and most recent dates in that column
    
    :param df: the dataframe you want to print the min and max dates from
    :param date_col: the name of the column that contains the date
    :type date_col: str
    :param type_date: str = the type of date you're looking for, e.g. "transaction" or "order"
    :type type_date: str
    :return: The earliest date and the most recent date for the date column in the dataframe.
    """
    if df is not None:
        return print(f" The earliest {type_date} date is: {df[date_col].min()}, and the most recent {type_date} date is {df[date_col].max()}") 
    
def print_num_rows_cols(df, df_name: str):
    """
    This function takes in a dataframe and a string, and prints out the number of rows and columns in
    the dataframe
    
    :param df: the dataframe you want to print the number of rows and columns for
    :param df_name: str = name of the dataframe
    :type df_name: str
    :return: The number of rows and columns in the dataframe.
    """
    if df is not None:
        return print(f"{df_name} data has {df.shape[1]} columns and {df.shape[0]} rows.")
        
def check_null_values(df):
    """
    It takes a dataframe as input and returns a dataframe with the number of missing values and the
    percentage of missing values for each column
    
    :param df: The dataframe to check for null values
    :return: A dataframe with the columns that have missing values and the percentage of missing values.
    """
    if df is not None:
        try:
            missing_value = df.isnull().sum()
            mis_val_percent = 100 * df.isnull().sum() / len(df)
            mis_val_table = pd.concat([missing_value, mis_val_percent], axis=1)
            mis_val_table_rename_cols = mis_val_table.rename(
                columns={0: "Missing Values", 1: "% of Total Values"}
            )
            mis_val_table_rename_cols = (
                mis_val_table_rename_cols[mis_val_table_rename_cols.iloc[:, 1] != 0]
                .sort_values("% of Total Values", ascending=False)
                .round(1)
            )
            print(
                "There are "
                + str(mis_val_table_rename_cols.shape[0])
                + " columns that have missing values"
            )
            return mis_val_table_rename_cols
        except Exception as e:
            print(f"Exception on checking missing values with error: {e}")
            
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

def kpss_test(timeseries):
    """
    The KPSS test is a test for stationarity. It tests the null hypothesis that the time series is trend
    stationary. The KPSS test is a non-parametric test. It does not assume anything about the
    distribution of the data
    
    :param timeseries: the time series you want to test
    """
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

def individual_train_and_forecast(df,code:str):  # Initiate the model
  m = Prophet(growth='linear',
                  yearly_seasonality=False,
                  weekly_seasonality=False,
                  daily_seasonality=False,
                  seasonality_mode='multiplicative',
                  seasonality_prior_scale=10,
                  changepoint_prior_scale=.05
                 ).add_seasonality(name='yearly',
                                    period=365.25,
                                    fourier_order=4,
                                    prior_scale=10,
                                    mode='additive')
  
  # Fit the model
  m.fit(df[df['AdminCode'] == code])  # Make predictions
  future = m.make_future_dataframe(periods=31, freq='D')
  forecast_values = m.predict(future)
  plotting_figure = m.plot(forecast_values)
  a = add_changepoints_to_plot(plotting_figure.gca(), m, forecast_values)
  plt.show()

def train_and_forecast(group):  # Initiate the model
  m = Prophet(growth='linear',
                  yearly_seasonality=False,
                  weekly_seasonality=False,
                  daily_seasonality=False,
                  seasonality_mode='multiplicative',
                  seasonality_prior_scale=10,
                  changepoint_prior_scale=.05
                 ).add_seasonality(name='yearly',
                                    period=365.25,
                                    fourier_order=4,
                                    prior_scale=10,
                                    mode='additive')
    
def train(features, target, method):
    """
    The function takes in a dataframe of features, a dataframe of target, and a method. It then fits the
    method to the features and target and returns the model
    
    :param features: the dataframe of features
    :param target: The target variable
    :param method: The method you want to use to train the model
    :return: The model is being returned.
    """
    model = method.fit(features, target)
    return model

def test(model, test_features, test_target):
    """
    The function takes in a model, test features, and test target, and returns the RMSE value
    
    :param model: the model you want to test
    :param test_features: the features of the test set
    :param test_target: the actual values of the target variable
    :return: The mean squared error of the model.
    """
    rmse = []
    predictions = model.predict(test_features)
    mse = mean_squared_error(test_target, predictions)
    mse_val = np.sqrt(mse)
    rmse.append(mse_val)
    return rmse

def times_series_cross_validate(attributes, target, method):
    """
    The function takes in the attributes and target dataframes, and the method of classification, and
    returns a list of accuracies for each fold
    
    :param attributes: the dataframe of features
    :param target: The target variable
    :param method: the method to use for training the model. This can be either 'logistic' or 'svm'
    :return: The accuracies of each fold.
    """
    fold_accuracies = []
    #kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    tscv = TimeSeriesSplit(n_splits=5, test_size=2)
    for train_index, test_index in tscv.split(attributes):
        train_features, test_features = attributes.iloc[train_index], attributes.iloc[test_index]
        test_labels, train_labels = target.iloc[test_index], target.iloc[train_index]
        model = train(train_features, train_labels, method)
        accuracy = test(model, test_features, test_labels)
        fold_accuracies.append(accuracy)
    return fold_accuracies