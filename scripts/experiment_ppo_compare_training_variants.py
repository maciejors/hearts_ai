from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_both_agents, train_playing_agent

reward_setting = 'dense'
best_card_passing_agent = MaskablePPO.load(
    'output/logs/ppo_passing_compare_stages_setups/every_512x5/run_9/eval_rule_based/best_model'
)

for i in range(10):
    train_both_agents(
        agent_cls_play=MaskablePPO,
        agent_cls_card_pass=MaskablePPO,
        play_env_kwargs={'reward_setting': reward_setting},
        card_pass_env_kwargs={},
        log_path='output/logs/ppo_compare_training_variants/both_20',
        stages_lengths_episodes=[192 * 100] * 15,
        swap_period_multipl=20,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )
    train_both_agents(
        agent_cls_play=MaskablePPO,
        agent_cls_card_pass=MaskablePPO,
        play_env_kwargs={'reward_setting': reward_setting},
        card_pass_env_kwargs={},
        log_path='output/logs/ppo_compare_training_variants/both_50',
        stages_lengths_episodes=[192 * 100] * 15,
        swap_period_multipl=50,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )
    train_playing_agent(
        agent_cls=MaskablePPO,
        eval_card_passing_agent=best_card_passing_agent,
        env_kwargs={'reward_setting': reward_setting},
        log_path='output/logs/ppo_compare_training_variants/separate',
        stages_lengths_episodes=[192 * 100] * 15,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )
