from sb3_contrib import MaskablePPO

from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.training import train_both_agents

train_both_agents(
    agent_cls_play=MaskablePPO,
    agent_cls_card_pass=MaskablePPO,
    play_env_kwargs={'reward_setting': 'dense'},
    card_pass_env_kwargs={},
    log_path='output/logs/final_training/ppo_both',
    stages_lengths_episodes=[192 * 100] * 40,
    swap_period_multipl=20,
    eval_freq_episodes=50000,
    n_eval_episodes=10000,
    progress_bar=True,
    random_state=3,
)
train_both_agents(
    agent_cls_play=MaskableMCTSRL,
    agent_cls_card_pass=MaskablePPO,
    play_env_kwargs={'reward_setting': 'sparse'},
    card_pass_env_kwargs={},
    log_path='output/logs/final_training/mctsrl_play_ppo_pass',
    stages_lengths_episodes=[192 * 100] * 40,
    swap_period_multipl=20,
    eval_freq_episodes=50000,
    n_eval_episodes=10000,
    progress_bar=True,
    #random_state=TBC,
)
