# 18 · Reinforcement learning

> Learning from consequences: values, policies, models and preferences.

RL is the study of sequential decisions under delayed reward. This folder builds the Bellman machinery first, then the deep algorithms, then the alignment-flavoured ones (offline RL, reward models, DPO) that dominate current practice.

<!-- catalog:begin -->
**18 atoms · 87 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · MDP solvers | [`mdp.py`](mdp.py) | 1 | 4 | Value iteration, policy iteration and convergence guarantees. |
| · Temporal-difference learning | [`td_learning.py`](td_learning.py) | 1 | 7 | TD(0), SARSA, Q-learning, n-step returns and traces. |
| · DQN | [`dqn.py`](dqn.py) | 1 | 7 | Replay, target networks and the improvements that mattered. |
| · REINFORCE | [`reinforce.py`](reinforce.py) | 1 | 4 | The policy gradient theorem, baselines and variance reduction. |
| · Actor-critic | [`actor_critic.py`](actor_critic.py) | 1 | 5 | A2C and generalised advantage estimation. |
| · PPO | [`ppo.py`](ppo.py) | 1 | 5 | Clipped surrogate objective and the trust-region intuition. |
| · SAC | [`sac.py`](sac.py) | 1 | 4 | Maximum-entropy RL with automatic temperature tuning. |
| · Bandits | [`bandits.py`](bandits.py) | 1 | 7 | Exploration versus exploitation, from UCB to LinUCB. |
| · TRPO | [`trpo.py`](trpo.py) | 2 | 3 | Natural gradient, conjugate gradient and line search. |
| · DDPG | [`ddpg.py`](ddpg.py) | 2 | 3 | Deterministic policy gradients with target smoothing. |
| · TD3 | [`td3.py`](td3.py) | 2 | 4 | Twin critics and delayed policy updates. |
| · Model-based RL | [`model_based.py`](model_based.py) | 2 | 4 | Dyna, learned dynamics and model-predictive control. |
| · AlphaZero | [`alphazero.py`](alphazero.py) | 2 | 4 | Monte Carlo tree search with policy/value networks and self-play. |
| · Imitation learning | [`imitation.py`](imitation.py) | 2 | 4 | Behaviour cloning, DAgger and GAIL. |
| · Reward modelling & DPO | [`reward_modeling.py`](reward_modeling.py) | 2 | 7 | RLHF, preference models and direct preference optimisation. |
| · Offline RL | [`offline_rl.py`](offline_rl.py) | 3 | 5 | CQL and IQL: learn without exploration. |
| · Exploration bonuses | [`exploration.py`](exploration.py) | 3 | 6 | Count-based, RND and curiosity-driven exploration. |
| · Multi-agent RL | [`multi_agent.py`](multi_agent.py) | 3 | 4 | Self-play, MADDPG and value factorisation. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **MDP solvers** ([`mdp.py`](mdp.py) · [mdp.md](mdp.md)) — `value_iteration` · `policy_iteration` · `modified_policy_iteration` · `linear_programming`
- **Temporal-difference learning** ([`td_learning.py`](td_learning.py) · [td_learning.md](td_learning.md)) — `td0` · `sarsa` · `q_learning` · `expected_sarsa` · `n_step` · `sarsa_lambda` · `q_lambda`
- **DQN** ([`dqn.py`](dqn.py) · [dqn.md](dqn.md)) — `dqn` · `double_dqn` · `dueling` · `prioritized` · `n_step_dqn` · `c51` · `rainbow_lite`
- **REINFORCE** ([`reinforce.py`](reinforce.py) · [reinforce.md](reinforce.md)) — `vanilla` · `baseline` · `returns_normalized` · `entropy_bonus`
- **Actor-critic** ([`actor_critic.py`](actor_critic.py) · [actor_critic.md](actor_critic.md)) — `a2c` · `a3c` · `gae` · `td_ac` · `n_step_ac`
- **PPO** ([`ppo.py`](ppo.py) · [ppo.md](ppo.md)) — `clip` · `kl_penalty` · `adaptive_kl` · `recurrent_ppo` · `ppo_lagrangian`
- **SAC** ([`sac.py`](sac.py) · [sac.md](sac.md)) — `sac` · `sac_discrete` · `automatic_temperature` · `twin_q`
- **Bandits** ([`bandits.py`](bandits.py) · [bandits.md](bandits.md)) — `epsilon_greedy` · `ucb1` · `ucb_improved` · `thompson` · `linucb` · `exp3` · `contextual`
- **TRPO** ([`trpo.py`](trpo.py) · [trpo.md](trpo.md)) — `trpo` · `natural_pg` · `line_search`
- **DDPG** ([`ddpg.py`](ddpg.py) · [ddpg.md](ddpg.md)) — `ddpg` · `target_smoothing` · `ou_noise`
- **TD3** ([`td3.py`](td3.py) · [td3.md](td3.md)) — `td3` · `twin_critic` · `delayed_update` · `target_policy_smoothing`
- **Model-based RL** ([`model_based.py`](model_based.py) · [model_based.md](model_based.md)) — `dyna` · `learned_model` · `mpc` · `mopo`
- **AlphaZero** ([`alphazero.py`](alphazero.py) · [alphazero.md](alphazero.md)) — `alphazero` · `muzero` · `puct` · `self_play`
- **Imitation learning** ([`imitation.py`](imitation.py) · [imitation.md](imitation.md)) — `behavior_cloning` · `dagger` · `gail` · `sqil`
- **Reward modelling & DPO** ([`reward_modeling.py`](reward_modeling.py) · [reward_modeling.md](reward_modeling.md)) — `rlhf` · `dpo` · `ipo` · `kto` · `orpo` · `reward_model` · `ppo_rlhf`
- **Offline RL** ([`offline_rl.py`](offline_rl.py) · [offline_rl.md](offline_rl.md)) — `cql` · `iql` · `bcq` · `td3_bc` · `awac`
- **Exploration bonuses** ([`exploration.py`](exploration.py) · [exploration.md](exploration.md)) — `count_based` · `rnd` · `curiosity` · `icm` · `entropy_bonus` · `noise`
- **Multi-agent RL** ([`multi_agent.py`](multi_agent.py) · [multi_agent.md](multi_agent.md)) — `self_play` · `maddpg` · `qmix` · `vdn`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
