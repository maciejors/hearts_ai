from sb3_contrib import MaskablePPO

from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.training import train_both_agents

train_both_agents(
    agent_cls_play=MaskablePPO,
    agent_cls_card_pass=MaskablePPO,
    play_env_kwargs={'reward_setting': 'dense'},
    card_pass_env_kwargs={},
    log_path='output/logs/final_training/ppo_full',
    stages_lengths_episodes=[192 * 100] * 30,
    swap_period_multipl=20,
    eval_freq_episodes=20000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=3,
)
train_both_agents(
    agent_cls_play=MaskablePPO,
    agent_cls_card_pass=MaskablePPO,
    play_env_kwargs={'reward_setting': 'dense', 'observation_setting': 'compact'},
    card_pass_env_kwargs={'play_env_obs_settings': 'compact'},
    log_path='output/logs/final_training/ppo_compact',
    stages_lengths_episodes=[192 * 100] * 30,
    swap_period_multipl=20,
    eval_freq_episodes=20000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=4,
)
train_both_agents(
    agent_cls_play=MaskableMCTSRL,
    agent_cls_card_pass=MaskablePPO,
    play_env_kwargs={'reward_setting': 'sparse'},
    card_pass_env_kwargs={},
    log_path='output/logs/final_training/mctsrl_full',
    stages_lengths_episodes=[192 * 100] * 30,
    swap_period_multipl=20,
    eval_freq_episodes=20000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=2,
)
train_both_agents(
    agent_cls_play=MaskableMCTSRL,
    agent_cls_card_pass=MaskablePPO,
    play_env_kwargs={'reward_setting': 'sparse', 'observation_setting': 'compact'},
    card_pass_env_kwargs={'play_env_obs_settings': 'compact'},
    log_path='output/logs/final_training/mctsrl_compact',
    stages_lengths_episodes=[192 * 100] * 30,
    swap_period_multipl=20,
    eval_freq_episodes=20000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=2,
)
