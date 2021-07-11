import gym
import numpy as np
import pandas as pd

from os import path, makedirs
from gym import spaces
from gym.utils import seeding

from random import randrange
from datetime import time

from dateutil.relativedelta import *
from sklearn.preprocessing import minmax_scale

from forex_bot.utils.env_funct import terminal, close_trade
from forex_bot.utils.reward_funct import score_1
from forex_bot.utils.four_act_funct import execute_action, next_obs


class ForexEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 mode: str = 'train_ts',
                 timeframe: str = 'M30',
                 model_name: str = 'forex_model',
                 reward: str = 'score-test',
                 lookback: int = 10,
                 data=None,
                 commission: bool = False,
                 start_balance: float = 10000.0,
                 leverage: int = 100,
                 risk: float = 10.0,
                 max_loss: float = 2.5,
                 test_eps: int = 4,
                 ):
        super(ForexEnv, self).__init__()

        assert data is not None, 'Provide data to start!'
        assert isinstance(data, list)
        assert mode in ['train_ts', 'train_rand', 'test'], 'Invalid mode selected! Possible options: train, test'

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(spaces={
            'ohlcv': spaces.Box(low=-1, high=1, shape=(lookback, 9), dtype=np.float32),
            'wallet': spaces.Box(low=-1, high=1, shape=(lookback, 6), dtype=np.float32),
            # 'ohlcv_wallet': spaces.Box(low=-1, high=1, shape=(15, lookback), dtype=np.float32),
        })
        self.seed()

        tf_spec_dict = {
            'M15': {'week_steps': 480, 'ep_steps': 480 - lookback,
                    'close_time': time(23, 45), 't_delta': relativedelta(minutes=15)},
            'M30': {'week_steps': 240, 'ep_steps': 240 - lookback,
                    'close_time': time(23, 30), 't_delta': relativedelta(minutes=30)},
            'H1': {'week_steps': 120, 'ep_steps': 120 - lookback,
                   'close_time': time(23, 00), 't_delta': relativedelta(hours=1)},
        }

        mode_data_specs = {
            'train_ts': {'reset_index': data[0].shape[0]/tf_spec_dict[timeframe]['week_steps'], 'data_index': 0},
            'train_rand': {'reset_index': len(data), 'data_index': 0},
            'test': {'reset_index': len(data), 'data_index': 0},
        }

        self.model_specs = {
            'mode': mode,
            'model_name': model_name,
            'start_balance': start_balance,
            'lookback': lookback,
            'risk': risk/100,
            'max_loss_percent': 1 - (max_loss/100),
            'reward': reward,
            'leverage': leverage,
            'commission': commission,
            'tf_specs': tf_spec_dict.get(timeframe),
            'data_specs': mode_data_specs.get(mode),
            'test_eps': test_eps
        }

        self.ep_specs = {
            'step': 0,
            'max_steps': 0,
            'total_trades': 0,
            'bad_trades': 0,
            'max_balance': 0.0,
            'max_loss': 0.0,
            'reward_list': [],
            'ep_start_date': '',
            'ep_end_date': ''
        }

        self.all_data = data
        self.active_df = np.array([])
        self.wallet_info = np.array([])
        self.trade_info = np.array([])

        if mode == 'test':
            single_run_cols = ['model_name', 'subset_data', 'balance', 'min_balance', 'max_balance', 'current_step',
                               'episode_steps', 'total_trades', 'bad_trades', 'good_trades', 'long_t', 'short_t',
                               'profit_factor', 'worst_trade', 'best_trade', 'reward', 'reason']

            summary_cols = ['model_name', 'subset_data', 'train_steps', 'avg_b', 'min_b', 'max_b', 'avg_step',
                            'episode_steps', 'run_ratio', 'dist_10', 'dist_25', 'dist_50', 'dist_75', 'dist_90',
                            'total_trades', 'bad_trades', 'good_trades', 'long_t', 'short_t', 'profit_factor',
                            'worst_trade', 'best_trade', 'avg_r', 'min_r', 'max_r', 'total_avg_b', 'total_avg_r',
                            'reason']

            self.test_results = pd.DataFrame(columns=single_run_cols)
            self.test_summary = pd.DataFrame(columns=summary_cols)
            self.train_steps = 0
            self.test_freq = 0.1

            root_dir = path.dirname(path.abspath(__file__))
            test_path = path.join(root_dir.split('forex_bot')[0], 'test_results')
            if not path.isdir(test_path):
                makedirs(test_path)

            self.test_result_path = path.join(test_path, f'{model_name}.csv')
            if not path.isfile(self.test_result_path):
                self.test_summary.to_csv(self.test_result_path, sep=',', index=False, encoding='UTF-8')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        mode = self.model_specs['mode']

        week_steps = self.model_specs['tf_specs']['week_steps']
        data_index = self.model_specs['data_specs']['data_index']
        reset_index = self.model_specs['data_specs']['reset_index']

        if mode == 'train_ts':
            start = data_index * week_steps
            self.active_df = self.all_data[0][start:start+week_steps]

        elif mode == 'train_rand':
            current_pair = self.all_data[data_index]
            rand_start = randrange(0, len(current_pair), week_steps)
            self.active_df = current_pair[rand_start:rand_start+week_steps]

        else:
            self.active_df = self.all_data[data_index]

        data_index += 1
        if data_index == reset_index:
            data_index = 0
        self.model_specs['data_specs']['data_index'] = data_index

        ep_start_d = self.active_df[0][2]
        ep_end_d = self.active_df[-1][2]

        assert ep_start_d.isoweekday() == 1 and ep_start_d.time() == time(0, 0)
        assert ep_end_d.isoweekday() == 5 and ep_end_d.time() == self.model_specs['tf_specs']['close_time']

        self.ep_specs = {
            'step': 0,
            'max_steps': self.model_specs['tf_specs']['ep_steps'],
            'total_trades': 0,
            'bad_trades': 0,
            'max_balance': 0.0,
            'max_loss': 0.0,
            'reward_list': [],
            'ep_start_date': str(ep_start_d),
            'ep_end_date': str(ep_end_d)
        }

        start_balance, lookback = self.model_specs['start_balance'], self.model_specs['lookback']

        # WALLET INFO: step, position, balance, equity, pos_value, pos_return
        self.wallet_info = np.repeat([[0.0, 0.0, start_balance, start_balance, 0.0, 0.0]], repeats=lookback, axis=0)
        # TRADE INFO:
        # trade_nr, time, balance_before, position, size, open_p, close_p, pos_return, reason
        self.trade_info = np.empty(shape=(0, 9), dtype=object)
        reset_obs = {
            'ohlcv': minmax_scale(np.delete(self.active_df[:lookback], np.s_[0:3], axis=1), feature_range=(-1, 1)),
            'wallet': minmax_scale(self.wallet_info, feature_range=(-1, 1))
        }

        return reset_obs

    def step(self, action):
        ohlcv_values = self._current_ohlcv()

        self.wallet_info, self.trade_info, self.ep_specs = execute_action(
            self.wallet_info, self.trade_info, ohlcv_values, self.model_specs, self.ep_specs, action)

        reward = score_1(self.wallet_info[-1][5], self.model_specs['start_balance'])
        self.ep_specs['reward_list'].append(reward)

        new_obs, self.wallet_info = next_obs(self.active_df, self.wallet_info, self.trade_info,
                                             self.model_specs, self.ep_specs)

        done, reason = terminal(self.ep_specs, self.wallet_info[-1][3])

        if done:
            if int(self.wallet_info[-1][1]) != 0:
                position, balance, equity, trade_return, self.ep_specs = close_trade(
                    self.trade_info, self.wallet_info, self._current_ohlcv(),
                    self.ep_specs, self.model_specs['commission'])
                self.wallet_info[-1] = [self.ep_specs['step'], position, balance, equity, 0.0, trade_return]

                reward = score_1(self.wallet_info[-1][5], self.model_specs['start_balance'])
                self.ep_specs['reward_list'].append(reward)

            if self.model_specs['mode'] == 'test':
                self._to_csv(reason)

        return new_obs, reward, done, {}

    def _current_ohlcv(self):
        row = self.model_specs['lookback'] + self.ep_specs['step'] - 1

        current_ohlcv = {
            'real_time': str(self.active_df[row][2] + self.model_specs['tf_specs']['t_delta']),
            'ohlcv_time': self.active_df[row][2],
            'high': round(self.active_df[row][4], 5),
            'low': round(self.active_df[row][5], 5),
            'buy_p': round(self.active_df[row][7], 5),
            'sell_p': round(self.active_df[row][8], 5),
        }

        return current_ohlcv

    def _to_csv(self, reason: str):
        wallet_info = pd.DataFrame(self.wallet_info,
                                   columns=['step', 'position', 'balance', 'equity', 'trade_value', 'trade_return'])
        trade_info = pd.DataFrame(self.trade_info,
                                  columns=['trade_nr', 'time', 'balance', 'position', 'trade_size', 'open_p',
                                           'close_p', 'trade_return', 'reason'])
        wallet_info = wallet_info[self.model_specs['lookback']:]

        start_balance = self.model_specs['start_balance']
        balance = round(((wallet_info['balance'].values[-1] / start_balance) - 1) * 100, 4)
        min_b = round(((wallet_info['balance'].min() / start_balance) - 1) * 100, 4)
        max_b = round(((wallet_info['balance'].max() / start_balance) - 1) * 100, 4)

        total_trades = trade_info['trade_nr'].values[-1] if len(trade_info) > 0 else 0
        # if balance == 0.0 and total_trades == 0:
        #     balance = -100

        trade_return = trade_info['trade_return']
        neg_trades = trade_return.loc[trade_return.astype(float) < 0.0].to_list()
        pos_trades = trade_return.loc[trade_return.astype(float) >= 0.0].to_list()
        bad_trades = len(neg_trades)
        good_trades = len(pos_trades)

        step = wallet_info['step'].values[-1]

        if sum(neg_trades) != 0.0 and sum(pos_trades) != 0.0:
            profit_factor = round(sum(pos_trades) / -sum(neg_trades), 4)
        else:
            profit_factor = 0.0

        worst_trade = round(min(trade_return) / start_balance * 100, 4) if len(trade_info) > 0 else 0
        best_trade = round(max(trade_return) / start_balance * 100, 4) if len(trade_info) > 0 else 0
        trade_pos = trade_info['position']
        long_trades = len(trade_pos.loc[trade_pos.astype(int) == 1].to_list()) if len(trade_info) > 0 else 0
        short_trades = len(trade_pos.loc[trade_pos.astype(int) == -1].to_list()) if len(trade_info) > 0 else 0
        reward = sum(self.ep_specs['reward_list'])

        single_run = dict(model_name=self.model_specs['model_name'], subset_data=self.ep_specs['ep_start_date'],
                          balance=balance, min_balance=min_b, max_balance=max_b, current_step=step,
                          episode_steps=self.model_specs['tf_specs']['ep_steps'], total_trades=total_trades,
                          bad_trades=bad_trades, good_trades=good_trades, long_t=long_trades, short_t=short_trades,
                          profit_factor=profit_factor, worst_trade=worst_trade, best_trade=best_trade, reward=reward,
                          reason=reason)
        self.test_results = self.test_results.append(single_run, ignore_index=True)

        if len(self.test_results) == 10:
            dtypes = dict(balance='float32', min_balance='float32', max_balance='float32', current_step='int',
                          episode_steps='int', total_trades='int', bad_trades='int', good_trades='int',
                          long_t='int', short_t='int', profit_factor='float32',
                          worst_trade='float32', best_trade='float32', reward='float32')
            self.test_results = self.test_results.astype(dtype=dtypes)
            summary = self.test_results.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

            good_runs = self.test_results['balance'].loc[self.test_results['balance'] > 0.0].to_list()
            run_ratio = round(len(good_runs) / self.model_specs['test_eps'], 4)

            test_avg = dict(model_name=self.model_specs['model_name'], subset_data=self.ep_specs['ep_start_date'],
                            train_steps=self.train_steps,
                            avg_b=summary['balance']['mean'],
                            min_b=summary['balance']['min'],
                            max_b=summary['balance']['max'],
                            avg_step=summary['current_step']['mean'],
                            episode_steps=self.model_specs['tf_specs']['ep_steps'],
                            run_ratio=run_ratio,
                            dist_10=summary['balance']['10%'],
                            dist_25=summary['balance']['25%'],
                            dist_50=summary['balance']['50%'],
                            dist_75=summary['balance']['75%'],
                            dist_90=summary['balance']['90%'],
                            total_trades=summary['total_trades']['mean'],
                            bad_trades=summary['bad_trades']['mean'],
                            good_trades=summary['good_trades']['mean'],
                            long_t=summary['long_t']['mean'],
                            short_t=summary['short_t']['mean'],
                            profit_factor=summary['profit_factor']['mean'],
                            worst_trade=summary['worst_trade']['mean'],
                            best_trade=summary['best_trade']['mean'],
                            avg_r=summary['reward']['mean'],
                            min_r=summary['reward']['min'],
                            max_r=summary['reward']['max'],
                            total_avg_b=0.0,
                            total_avg_r=0.0,
                            reason=self.test_results['reason'].value_counts().idxmax(),
                            )
            self.test_summary = self.test_summary.append(test_avg, ignore_index=True)
            self.test_results = self.test_results[0:0]

            if len(self.test_summary) == 4:
                self.test_summary['total_avg_b'] = round(self.test_summary['avg_b'].mean(), 4)
                self.test_summary['total_avg_r'] = round(self.test_summary['avg_r'].mean(), 4)
                self.test_summary = self.test_summary.round(4)
                print(self.test_summary)
                self.test_summary.to_csv(path_or_buf=self.test_result_path,
                                         sep=',', index=False, encoding='UTF-8', mode='a', header=None)

                self.train_steps = round(self.train_steps + self.test_freq, 4)
                self.test_summary = self.test_summary[0:0]
            else:
                pass
        else:
            pass

    def render(self, mode='human'):
        pass
