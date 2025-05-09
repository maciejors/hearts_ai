from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_playing_agent

train_playing_agent(
    agent_cls=MaskablePPO,
    env_kwargs={'reward_setting': 'dense'},
    log_path='output/logs/ppo_compare_reward_settings/dense',
    stages_lengths_episodes=[192 * 500] * 4,
    eval_freq_episodes=10000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=28,
)
train_playing_agent(
    agent_cls=MaskablePPO,
    env_kwargs={'reward_setting': 'sparse'},
    log_path='output/logs/ppo_compare_reward_settings/sparse',
    stages_lengths_episodes=[192 * 500] * 4,
    eval_freq_episodes=10000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=28,
)
