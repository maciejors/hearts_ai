from sb3_contrib import MaskablePPO

from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.training import train_both_agents

reward_setting_mctsrl = 'sparse'

for i in range(1, 5):
    train_both_agents(
        agent_cls_play=MaskableMCTSRL,
        agent_cls_card_pass=MaskablePPO,
        play_env_kwargs={'reward_setting': reward_setting_mctsrl},
        card_pass_env_kwargs={},
        log_path='output/logs/mctsrl/mctsrl_play_ppo_pass',
        stages_lengths_episodes=[192 * 100] * 5,
        swap_period_multipl=20,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )
