from sb3_contrib.ppo_mask import MaskablePPO

from hearts_ai.rl.training import train_card_passing_agent

best_playing_agent = MaskablePPO.load(
    'output/logs/ppo_playing_compare_stages_setups/every_192x1000/eval/best_model'
)
train_card_passing_agent(
    agent_cls=MaskablePPO,
    playing_agent=best_playing_agent,
    env_kwargs={},
    log_path='output/logs/ppo_passing_compare_stages_setups/every_512x5',
    stages_lengths_episodes=[512 * 5] * 20,
    progress_bar=True,
    random_state=28,
)
train_card_passing_agent(
    agent_cls=MaskablePPO,
    playing_agent=best_playing_agent,
    env_kwargs={},
    log_path='output/logs/ppo_passing_compare_stages_setups/every_512x20',
    stages_lengths_episodes=[512 * 20] * 5,
    progress_bar=True,
    random_state=28,
)
train_card_passing_agent(
    agent_cls=MaskablePPO,
    playing_agent=best_playing_agent,
    env_kwargs={},
    log_path='output/logs/ppo_passing_compare_stages_setups/every_512x50',
    stages_lengths_episodes=[512 * 50] * 2,
    progress_bar=True,
    random_state=28,
)
