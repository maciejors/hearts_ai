from sb3_contrib import MaskablePPO

from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.training import train_both_agents

reward_setting_mctsrl = 'sparse'
reward_setting_ppo = 'dense'

for i in range(5):
    train_both_agents(
        agent_cls_play=MaskableMCTSRL,
        agent_cls_card_pass=MaskablePPO,
        play_env_kwargs={'reward_setting': reward_setting_mctsrl, 'observation_setting': 'full'},
        card_pass_env_kwargs={},
        log_path='output/logs/obs_types/mctsrl_obs_full',
        stages_lengths_episodes=[192 * 100] * 5,
        swap_period_multipl=20,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )
    train_both_agents(
        agent_cls_play=MaskableMCTSRL,
        agent_cls_card_pass=MaskablePPO,
        play_env_kwargs={'reward_setting': reward_setting_mctsrl, 'observation_setting': 'compact'},
        card_pass_env_kwargs={'play_env_obs_settings': ['compact', 'full', 'full', 'full']},
        log_path='output/logs/obs_types/mctsrl_obs_compact',
        stages_lengths_episodes=[192 * 100] * 5,
        swap_period_multipl=20,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )

for i in range(10):
    train_both_agents(
        agent_cls_play=MaskablePPO,
        agent_cls_card_pass=MaskablePPO,
        play_env_kwargs={'reward_setting': reward_setting_ppo, 'observation_setting': 'compact'},
        card_pass_env_kwargs={'play_env_obs_settings': ['compact', 'full', 'full', 'full']},
        log_path='output/logs/obs_types/ppo_obs_compact',
        stages_lengths_episodes=[192 * 100] * 15,
        swap_period_multipl=20,
        eval_freq_episodes=10000,
        n_eval_episodes=10000,
        progress_bar=True,
        random_state=i,
    )
