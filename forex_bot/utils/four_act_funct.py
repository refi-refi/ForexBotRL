import numpy as np
import pandas as pd

from time import sleep
from datetime import datetime
from enum import Enum
from dateutil.parser import *
from forex_bot.utils.env_funct import close_trade

from sklearn.preprocessing import minmax_scale


class Actions(Enum):
    NoTrade = 0
    Long = 1
    Short = 2
    ExitTrade = 3


def execute_action(wallet_info: np.array, trade_info: np.array,
                   ohlcv_values: {}, model_params: {}, episodes_specs: {}, action):

    position = int(wallet_info[-1][1])
    balance = round(wallet_info[-1][2], 2)
    equity = round(wallet_info[-1][3], 2)
    trade_value, trade_return = .0, .0

    action: Actions = Actions(action % 4)

    if position == 0:
        if action in [Actions.NoTrade, Actions.ExitTrade]:
            pass
        else:

            balance_before = balance

            if action == Actions.Long:
                position = 1
                open_p = ohlcv_values['sell_p']

            else:
                assert action == Actions.Short
                position = -1
                open_p = ohlcv_values['buy_p']

            trade_size = round(model_params['start_balance'] * model_params['leverage'] * 0.1 / 1e5, 2)
            trade_nr = episodes_specs['total_trades']

            new_trade = np.array([
                trade_nr, ohlcv_values['real_time'], balance_before, position, trade_size, open_p, 0.0, 0.0, 'open'],
                dtype=object)
            trade_info = np.vstack((trade_info, new_trade))

    else:

        balance_before = round(trade_info[-1][2], 2)
        trade_size = round(trade_info[-1][4], 2)
        open_p = trade_info[-1][5]

        if action == Actions.ExitTrade:
            position, balance, equity, trade_return, episodes_specs = close_trade(
                trade_info, wallet_info, ohlcv_values, episodes_specs, model_params['commission'])

            if balance > episodes_specs['max_balance']:
                episodes_specs['max_balance'] = balance
                episodes_specs['max_loss'] = round(balance * model_params['max_loss_percent'], 2)

        else:
            if position == 1 and action in [Actions.Long, Actions.NoTrade]:
                close_p = ohlcv_values['buy_p']
                trade_value = round((trade_size * 1e5) * (1 - (open_p / close_p)), 2)
                equity = round(balance_before + trade_value, 2)

            else:
                assert position == -1 and action in [Actions.Short, Actions.NoTrade]
                close_p = ohlcv_values['sell_p']
                trade_value = round((trade_size * 1e5) * ((open_p / close_p) - 1), 2)
                equity = round(balance_before + trade_value, 2)

    wallet_change = np.array([episodes_specs['step'], position, balance, equity, trade_value, trade_return])
    wallet_info[-1] = wallet_change

    episodes_specs['step'] += 1

    return wallet_info, trade_info, episodes_specs


def next_obs(active_df: np.array, wallet_info: np.array, trade_info: np.array, model_specs: {}, episodes_specs: {}):

    position = int(wallet_info[-1][1])
    balance, equity = round(wallet_info[-1][2], 2), round(wallet_info[-1][3], 2)
    trade_value, trade_return = 0.0, 0.0

    current_obs = active_df[0 + episodes_specs['step']: model_specs['lookback'] + episodes_specs['step']]

    buy_p, sell_p = current_obs[-1][7], current_obs[-1][8]

    if position == 0:
        pass
    else:
        balance_before = round(trade_info[-1][2], 2)
        trade_amount = int(trade_info[-1][4] * 1e5)
        open_p = round(trade_info[-1][5], 5)

        if position == 1:
            trade_value = round(trade_amount * (1 - (open_p/buy_p)), 2)
            equity = round(balance_before + trade_value, 2)
        else:
            assert int(trade_info[-1][3]) == -1
            trade_value = round(trade_amount * ((open_p/sell_p) - 1), 2)
            equity = round(balance_before + trade_value, 2)

    new_wallet = [episodes_specs['step'], position, balance, equity, trade_value, trade_return]
    wallet_info = np.vstack((wallet_info, new_wallet))

    obs = {
        'ohlcv': minmax_scale(np.delete(current_obs, np.s_[0:3], axis=1), feature_range=(-1, 1)),
        'wallet': minmax_scale(wallet_info[-model_specs['lookback']:], feature_range=(-1, 1))
    }

    return obs, wallet_info
