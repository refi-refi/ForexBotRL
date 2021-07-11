import numpy as np


def close_trade(trade_info: np.array, wallet_info: np.array, ohlcv_values: {}, episodes_specs: {}, commission: bool):
    balance_before = trade_info[-1][2]
    trade_size = trade_info[-1][4]
    open_p = trade_info[-1][5]

    position = wallet_info[-1][1]

    if position == 1:
        close_p = ohlcv_values['buy_p']
        trade_return = round((trade_size * 1e5) * (1 - (open_p / close_p)), 2)
    else:
        close_p = ohlcv_values['sell_p']
        trade_return = round((trade_size * 1e5) * ((open_p / close_p) - 1), 2)

    if commission:
        trade_return = round(trade_return - (2.52 * trade_size), 2)

    equity = round(balance_before + trade_return, 2)
    balance = equity

    trade_info[-1][6] = close_p
    trade_info[-1][7] = trade_return
    trade_info[-1][8] = 'normal'
    position = 0
    episodes_specs['total_trades'] += 1

    if trade_return < 0.0:
        episodes_specs['bad_trades'] += 1
    else:
        episodes_specs['bad_trades'] = 0

    return position, balance, equity, trade_return, episodes_specs


def terminal(episodes_specs: {}, equity: float):
    key_list = ['step', 'max_steps', 'daily_loss', 'max_loss', 'bad_trades']

    step, max_steps, daily_loss, max_loss, bad_trades = (episodes_specs.get(key) for key in key_list)

    if step == max_steps:
        reason = 'No steps'
        done = True
    elif equity < max_loss:
        reason = 'Max loss'
        done = True
    elif bad_trades >= 5:
        reason = 'Bad trades'
        done = True
    else:
        reason = ''
        done = False

    return done, reason
