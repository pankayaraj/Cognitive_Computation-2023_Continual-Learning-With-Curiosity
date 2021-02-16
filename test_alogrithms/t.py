import gym

from stable_baselines3 import DDPG

env = gym.make('Pendulum-v0')

gym.register(
    id='SumoGUI-v0',
    entry_point='custom_envs.sumo:SUMOEnv_Initializer',
    max_episode_steps=1000,
    kwargs={'port_no': 8871}
)


env = gym.make('SumoGUI-v0')


model = DDPG('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
r = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

    r += reward
print(r)

env.close()





