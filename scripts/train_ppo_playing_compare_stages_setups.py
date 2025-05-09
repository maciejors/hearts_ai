from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_playing_agent

train_playing_agent(
    agent_cls=MaskablePPO,
    env_kwargs={'reward_setting': 'dense'},
    log_path='output/logs/ppo_compare_stages_setups/every_192x50',
    stages_lengths_episodes=[192 * 50] * 40,
    eval_freq_episodes=10000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=28,
)
train_playing_agent(
    agent_cls=MaskablePPO,
    env_kwargs={'reward_setting': 'dense'},
    log_path='output/logs/ppo_compare_stages_setups/every_192x200',
    stages_lengths_episodes=[192 * 200] * 10,
    eval_freq_episodes=10000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=28,
)
train_playing_agent(
    agent_cls=MaskablePPO,
    env_kwargs={'reward_setting': 'dense'},
    log_path='output/logs/ppo_compare_stages_setups/every_192x1000',
    stages_lengths_episodes=[192 * 1000] * 2,
    eval_freq_episodes=10000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=28,
)
