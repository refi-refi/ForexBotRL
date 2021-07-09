def score_1(trade_profit, start_balance):
    if trade_profit != 0.0:
        reward = trade_profit/start_balance * 100
    else:
        reward = 0

    return round(reward, 4)
