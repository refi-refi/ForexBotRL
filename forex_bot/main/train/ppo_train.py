from forex_bot.envs.forex_env_v0 import ForexEnv
from forex_bot.data.data_funct import load_data, agg_by_week, train_test_split

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


symbol = 'EURGBP'
timeframe = 'M30'
weekly_data = agg_by_week(load_data(symbol, timeframe), timeframe)
train_data, test_data = train_test_split(weekly_data)

train_env = make_vec_env(ForexEnv, env_kwargs=dict(data=train_data))
test_env = make_vec_env(ForexEnv, env_kwargs=dict(data=test_data, mode='test'))

eval_callback = EvalCallback(test_env, eval_freq=5000, deterministic=False, n_eval_episodes=40)

model = PPO("MultiInputPolicy", train_env, verbose=1)
model.learn(100000, callback=eval_callback)

