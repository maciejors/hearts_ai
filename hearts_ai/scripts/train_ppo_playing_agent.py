from stable_baselines3.common.env_checker import check_env

from hearts_ai.rl.env.play_env import HeartsPlayEnvironment

env = HeartsPlayEnvironment(opponents_callbacks=[lambda state: 0 for _ in range(3)], reward_setting='dense',
                            seed=10)
check_env(env)
