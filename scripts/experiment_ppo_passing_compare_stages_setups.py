from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_card_passing_agent

best_playing_agent = MaskablePPO.load(
    'output/logs/ppo_playing_compare_stages_setups/every_192x100/run_8/eval/best_model'
)
for i in range(10):
    train_card_passing_agent(
        agent_cls=MaskablePPO,
        playing_agent=best_playing_agent,
        env_kwargs={},
        log_path='output/logs/ppo_passing_compare_stages_setups/every_512x2',
        eval_freq_episodes=2000,
        n_eval_episodes=500,
        stages_lengths_episodes=[512 * 2] * 20,
        progress_bar=True,
        random_state=i,
    )
    train_card_passing_agent(
        agent_cls=MaskablePPO,
        playing_agent=best_playing_agent,
        env_kwargs={},
        log_path='output/logs/ppo_passing_compare_stages_setups/every_512x5',
        eval_freq_episodes=2000,
        n_eval_episodes=500,
        stages_lengths_episodes=[512 * 5] * 8,
        progress_bar=True,
        random_state=i,
    )
    train_card_passing_agent(
        agent_cls=MaskablePPO,
        playing_agent=best_playing_agent,
        env_kwargs={},
        log_path='output/logs/ppo_passing_compare_stages_setups/every_512x10',
        eval_freq_episodes=2000,
        n_eval_episodes=500,
        stages_lengths_episodes=[512 * 10] * 4,
        progress_bar=True,
        random_state=i,
    )
