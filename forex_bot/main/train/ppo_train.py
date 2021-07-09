from forex_bot.envs.forex_env_v0 import ForexEnv
from forex_bot.data.data_funct import train_data

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

train_data = train_data()

train_env = make_vec_env(ForexEnv, env_kwargs=dict(data=train_data))

model = PPO('MlpPolicy', train_env, verbose=1)
model.learn(10000)

