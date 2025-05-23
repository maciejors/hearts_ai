from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_both_agents

train_both_agents(
    agent_cls=MaskablePPO,
    play_env_kwargs={'reward_setting': 'dense'},
    card_pass_env_kwargs={},
    log_path='output/logs/ppo_both_agents/play',
    stages_lengths_episodes=[192 * 1000] * 2,
    eval_freq_episodes=10000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=28,
)
