from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


md_env = MechanismDesigner(5,20)
n_actions = md_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(MlpPolicy, md_env,learning_rate=3e-3 ,batch_size=10, buffer_size=100, gradient_steps=10,action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100000, log_interval=10)
model.save("outer_layer")
