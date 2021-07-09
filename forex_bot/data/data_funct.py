import os
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import acc_dist_index
from datetime import datetime, time
from dateutil.parser import *
from dateutil.relativedelta import *

pd.set_option('display.width', None)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_data(symbol: str = 'EURGBP',
               timeframe: str = 'M30',
               end_date: str = '2021-01',
               start_date: str = None) -> np.array:

    if start_date is not None:
        assert isinstance(start_date, str), 'start_date must be str!'
        assert parse(start_date) < parse(end_date), 'Start date must be < than end date!'

    mon_end_date = np.busday_offset(end_date, 0, roll='forward', weekmask='Mon')

    file_path = os.path.join(ROOT_DIR, f'{symbol}_{timeframe}.csv')

    df = pd.read_csv(file_path, low_memory=False, encoding='UTF-8')
    df['dt_dt'] = pd.to_datetime(df.date_time, format='%Y-%m-%d %H:%M:%S')

    if start_date is None:
        df = df.loc[df['dt_dt'] < mon_end_date]
    else:
        df = df.loc[(parse(start_date) <= df['dt_dt']) & (df['dt_dt'] < mon_end_date)]

    df = df.drop(columns=['dt_dt']).to_numpy()

    return [df]


# test = train_data(start_date='2020-12-01')
# print(test)


def test_data(symbol: str = 'EURGBP',
              timeframe: str = 'M30',
              test_month: str = '2021-01') -> []:

    steps_in_week = {
        'H1': 120,
        'M30': 240,
        'M15': 480
    }

    test_mondays = [np.busday_offset(test_month, i, roll='forward', weekmask='Mon') for i in range(4)]

    file_path = os.path.join(ROOT_DIR, f'{symbol}_{timeframe}.csv')

    df = pd.read_csv(file_path, low_memory=False, encoding='UTF-8')
    df['dt_dt'] = pd.to_datetime(df.date_time, format='%Y-%m-%d %H:%M:%S')

    all_test_week = []
    for monday in test_mondays:
        ep_data = df.loc[monday <= df['dt_dt']]
        ep_data = ep_data[:steps_in_week.get(timeframe)].drop(columns=['dt_dt'])
        all_test_week.append(ep_data.to_numpy())

    return all_test_week


# this = test_data()
# print(this)


def agg_by_week(df: pd.DataFrame, timeframe):

    timeframe_dict = {
        'M15': {'week_steps': 24*5*4, 'end_time': time(23, 45)},
        'M30': {'week_steps': 24*5*2, 'end_time': time(23, 30)},
        'H1': {'week_steps': 24*5, 'end_time': time(23, 00)},
    }
    tf_specs = timeframe_dict.get(timeframe)
    assert tf_specs is not None, 'Selected TF is not supported!'

    steps_in_week, end_time = tf_specs['week_steps'], tf_specs['end_time']

    df.date_time = pd.to_datetime(df.date_time)

    # df['week'] = ('year-' + df.date_time.dt.isocalendar().year.astype(str) + '_week-'
    #               + df.date_time.dt.isocalendar().week.astype(str))
    week_col = (df.date_time.dt.isocalendar().week.astype(str) + '-week_' +
                df.date_time.dt.isocalendar().year.astype(str))
    df.insert(3, 'week', week_col)

    week_counts = df['week'].value_counts()
    full_weeks = week_counts[week_counts == steps_in_week].index.values
    in_full = df['week'].isin(full_weeks)

    df = df[in_full].reset_index(drop=True).round(5)

    for i in np.arange(0, len(df), steps_in_week):
        start = df.date_time[i]
        end = df.date_time[i+steps_in_week-1]
        assert start.isoweekday() == 1 and start.time() == time(0, 0)
        assert end.isoweekday() == 5 and end.time() == end_time

    # save_path = f'data/{input_folder}/agg_by_week'
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    #
    # df.to_csv(f'{save_path}/{symbol}_{timeframe}.csv', index=False)

    return df


def time_cols(df):
    sin_day = round(np.sin((df['time'] - 4 * 24 * 60 * 60) * (2 * np.pi / (24 * 60 * 60))), 5)
    cos_day = round(np.cos((df['time'] - 4 * 24 * 60 * 60) * (2 * np.pi / (24 * 60 * 60))), 5)
    sin_week = round(np.sin((df['time'] - 4 * 24 * 60 * 60) * (2 * np.pi / (24 * 60 * 60 * 7))), 5)
    cos_week = round(np.cos((df['time'] - 4 * 24 * 60 * 60) * (2 * np.pi / (24 * 60 * 60 * 7))), 5)

    df.insert(3, 'cos_week', cos_week)
    df.insert(3, 'sin_week', sin_week)
    df.insert(3, 'cos_day', cos_day)
    df.insert(3, 'sin_day', sin_day)

    return df


def mavg(df: pd.DataFrame, col: str, n_ohlcv: int):
    df[f'MAVG_{col}_{n_ohlcv}'] = round(df[col].rolling(n_ohlcv).mean(), 5)

    return df.round(5)


def ema(df: pd.DataFrame, col: str, n_ohlcv: int):
    df[f'EMA_{col}_{n_ohlcv}'] = round(df[col].ewm(span=n_ohlcv, adjust=False).mean(), 5)

    return df.round(5)


def rsi(df: pd.DataFrame, column: str, n_ohlcv: int):
    ta_rsi = RSIIndicator(close=df[column], window=n_ohlcv)
    df[f'RSI_{n_ohlcv}'] = round(ta_rsi.rsi(), 2)

    return df.round(5)


def macd(df: pd.DataFrame):
    ta_macd = MACD(df['close'])
    df['MACD'] = ta_macd.macd()
    df['MACD_dif'] = ta_macd.macd_diff()

    return df.round(5)
