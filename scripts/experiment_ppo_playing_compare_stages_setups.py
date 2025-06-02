from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_playing_agent

reward_setting = 'dense'

for i in range(10):
    train_playing_agent(
        agent_cls=MaskablePPO,
        env_kwargs={'reward_setting': reward_setting},
        log_path='output/logs/ppo_playing_compare_stages_setups/every_192x10',
        stages_lengths_episodes=[192 * 10] * 300,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=28,
    )
    train_playing_agent(
        agent_cls=MaskablePPO,
        env_kwargs={'reward_setting': reward_setting},
        log_path='output/logs/ppo_playing_compare_stages_setups/every_192x100',
        stages_lengths_episodes=[192 * 100] * 30,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=28,
    )
    train_playing_agent(
        agent_cls=MaskablePPO,
        env_kwargs={'reward_setting': reward_setting},
        log_path='output/logs/ppo_playing_compare_stages_setups/every_192x1000',
        stages_lengths_episodes=[192 * 1000] * 3,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=28,
    )
