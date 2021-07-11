import os
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import time
from dateutil.parser import *

pd.set_option('display.width', None)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(symbol: str = 'EURGBP',
              timeframe: str = 'M30',
              start_date: str = None,
              end_date: str = '2021-02') -> pd.DataFrame:

    if start_date is not None:
        assert parse(start_date) < parse(end_date), 'Start date must be < than end date!'

    file_path = os.path.join(ROOT_DIR, f'{symbol}_{timeframe}.csv')
    df = pd.read_csv(file_path, low_memory=False, encoding='UTF-8', parse_dates=['date_time'], dayfirst=False)

    mon_end_date = np.busday_offset(end_date, 0, roll='forward', weekmask='Mon')
    if start_date is None:
        df = df.loc[df.date_time < mon_end_date]
    else:
        mon_start_date = np.busday_offset(start_date, 0, roll='forward', weekmask='Mon')
        df = df.loc[(mon_start_date <= df.date_time) & (df.date_time < mon_end_date)]

    return df.round(5)

def train_test_split(df: pd.DataFrame,
                     timeframe: str = 'M30',
                     split_month: str = '2021-01'):

    split_date = np.busday_offset(split_month, 0, roll='forward', weekmask='Mon')

    assert df.date_time[0] < split_date < df.date_time[len(df)-1], 'Split cannot be done, no data available!'

    steps_in_week = {
        'H1': 120,
        'M30': 240,
        'M15': 480
    }

    train_df = df.loc[df.date_time < split_date]
    train_dfs = [train_df.to_numpy()]

    test_mondays = [np.busday_offset(split_date, i, roll='forward', weekmask='Mon') for i in range(4)]

    test_dfs = []
    for monday in test_mondays:
        test_ep = df.loc[monday <= df.date_time]
        test_ep = test_ep[:steps_in_week.get(timeframe)]
        test_dfs.append(test_ep.to_numpy())

    return train_dfs, test_dfs


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

    df = df[in_full].reset_index(drop=True)

    for i in np.arange(0, len(df), steps_in_week):
        start = df.date_time[i]
        end = df.date_time[i+steps_in_week-1]
        assert start.isoweekday() == 1 and start.time() == time(0, 0)
        assert end.isoweekday() == 5 and end.time() == end_time

    df = df.drop(columns=['week']).round(5)

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
