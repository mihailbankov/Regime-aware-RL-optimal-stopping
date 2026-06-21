import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import deque
from pathlib import Path
from hmmlearn import hmm
from scipy.special import logsumexp

import torch
import torch.nn as nn
import torch.optim as optim
from pairs_trading_env_tc_newstate import PairsTradingEnv

# ── paste or import your env ──────────────────────────────────────────
# from pairs_trading_env import PairsTradingEnv
# If the file is in the same directory just uncomment the line above.
# Otherwise the full class is expected to be available as PairsTradingEnv.
# ─────────────────────────────────────────────────────────────────────

def dqn(df_a, df_b, SEED = 12):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_RATIO  = 0.80
    n = len(df_a)
    A = df_a['close'].values
    B = df_b['close'].values

    split        = int(n * TRAIN_RATIO)
    X = np.column_stack([np.ones(split), B[:split]])
    beta_ols, _ = np.linalg.lstsq(X, A[:split], rcond=None)[0], None
    alpha_ols = beta_ols[0]
    beta      = beta_ols[1]
    spread = A - beta * B


    df_a_train, df_b_train = df_a.iloc[:split], df_b.iloc[:split]
    df_a_test,  df_b_test  = df_a.iloc[split:].reset_index(drop=True), \
                            df_b.iloc[split:].reset_index(drop=True)
    

    MARKOV_STATES_NO = 3
    

    def beta_windows(a, b, window_size):
        a_s, b_s = pd.Series(a), pd.Series(b)
        roll = lambda s: s.rolling(window_size, min_periods=2)
        cov = roll(b_s).cov(a_s)          # cov(b, a)
        var = roll(b_s).var()              # var(b)
        betas = -(cov / var).to_numpy()
        betas = np.nan_to_num(betas, nan=-1.0, posinf=-1.0, neginf=-1.0)
        return betas
    betas = beta_windows(A, B, 50000)
    betas_train = betas[:split]
    betas_test = betas[split:]

    model = hmm.GaussianHMM(n_components=MARKOV_STATES_NO, covariance_type="full", n_iter=100)
    model.fit(betas_train.reshape(-1, 1))

    def forward_states(model, obs):
        log_emit  = model._compute_log_likelihood(obs)
        log_trans = np.log(model.transmat_)
        log_alpha = np.log(model.startprob_) + log_emit[0]
        states    = [int(np.argmax(log_alpha))]
        for t in range(1, len(obs)):
            log_alpha = log_emit[t] + logsumexp(log_alpha[:, None] + log_trans, axis=0)
            states.append(int(np.argmax(log_alpha)))
        return np.array(states)

    train_states = model.predict(betas_train.reshape(-1, 1))
    test_states  = forward_states(model, betas_test.reshape(-1, 1))

    # ── Hyper-parameters ──────────────────────────────────────────────────
    TRANSACTION_COST = 0.0001     # flat cost per open / close
    INITIAL_CASH     = 100.0

    GAMMA            = 0.995    # discount factor
    LR               = 1e-4    # learning rate
    BATCH_SIZE       = 64
    MEMORY_SIZE      = 50_000
    TARGET_UPDATE    = 200     # steps between target-net syncs
    N_EPISODES       = 400     # training episodes (each = full train series)

    EPS_START        = 1.00
    EPS_END          = 0.02
    EPS_DECAY        = 0.994   # per-episode multiplicative decay
    # ─────────────────────────────────────────────────────────────────────

    STATE_DIM  = PairsTradingEnv.STATE_DIM   # 4
    N_ACTIONS  = PairsTradingEnv.N_ACTIONS   # 3


    # ── Q-Network ─────────────────────────────────────────────────────────
    class QNet(nn.Module):
        def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),    nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    # ── Replay buffer ─────────────────────────────────────────────────────
    class ReplayBuffer:
        def __init__(self, capacity: int):
            self.buf = deque(maxlen=capacity)

        def push(self, s, a, r, s2, done):
            self.buf.append((s, a, r, s2, done))

        def sample(self, batch_size: int):
            batch = random.sample(self.buf, batch_size)
            s, a, r, s2, d = zip(*batch)
            return (
                torch.tensor(np.array(s),  dtype=torch.float32, device=DEVICE),
                torch.tensor(a,            dtype=torch.long,    device=DEVICE),
                torch.tensor(r,            dtype=torch.float32, device=DEVICE),
                torch.tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
                torch.tensor(d,            dtype=torch.float32, device=DEVICE),
            )

        def __len__(self): return len(self.buf)


    # ── DQN agent ─────────────────────────────────────────────────────────
    class DQNAgent:
        def __init__(self):
            self.policy_net = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
            self.target_net = QNet(STATE_DIM, N_ACTIONS).to(DEVICE)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
            self.memory    = ReplayBuffer(MEMORY_SIZE)
            self.steps     = 0
            self.epsilon   = EPS_START

        # ε-greedy selection
        def select_action(self, state: np.ndarray) -> int:
            if random.random() < self.epsilon:
                return random.randint(0, N_ACTIONS - 1)
            with torch.no_grad():
                t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                return int(self.policy_net(t).argmax(dim=1).item())

        def learn(self):
            if len(self.memory) < BATCH_SIZE:
                return

            s, a, r, s2, done = self.memory.sample(BATCH_SIZE)

            # Current Q values
            q_vals = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # Double-DQN target
            with torch.no_grad():
                best_actions = self.policy_net(s2).argmax(dim=1, keepdim=True)
                q_next       = self.target_net(s2).gather(1, best_actions).squeeze(1)
                target       = r + GAMMA * q_next * (1 - done)

            loss = nn.SmoothL1Loss()(q_vals, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            self.steps += 1
            if self.steps % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self):
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def run_episode(env: PairsTradingEnv, agents: list[DQNAgent], greedy: bool = True):
        """Run one episode; return per-step info dicts."""
        obs  = env.reset(is_test=True)
        done = False
        log  = []

        # Freeze epsilon for greedy evaluation
        saved_eps = agents[0].epsilon
        if greedy:
            for i in range(MARKOV_STATES_NO):
                agents[i].epsilon = 0.0

        while not done:
            state = test_states[env.current_step]
            action = agents[state].select_action(obs)
            obs, reward, done, info = env.step(action)
            info['action'] = action
            info['reward'] = reward
            log.append(info)

        for i in range(MARKOV_STATES_NO):
                agents[i].epsilon = saved_eps
        return pd.DataFrame(log)
    
    agents = []
    for i in range(MARKOV_STATES_NO):
        agents.append(DQNAgent())

    VALIDATION_START = int(split * 0.85)  # last 15% of train as validation

    val_env = PairsTradingEnv(
        df_a.iloc[VALIDATION_START:split].reset_index(drop=True),
        df_b.iloc[VALIDATION_START:split].reset_index(drop=True),
        beta=beta,
        operation_penalty=TRANSACTION_COST,
        initial_cash=INITIAL_CASH,
    )

    best_val_value = -np.inf
    best_state_dicts = None

    def make_train_env():
        return PairsTradingEnv(
            df_a_train[:VALIDATION_START], df_b_train[:VALIDATION_START],
            beta            = beta,
            operation_penalty = TRANSACTION_COST*10,
            initial_cash    = INITIAL_CASH,
        )

    train_env  = make_train_env()

    ep_returns = []

    for ep in range(1, N_EPISODES + 1):
        obs  = train_env.reset()
        done = False
        ep_reward = 0.0
        cnts = np.zeros(MARKOV_STATES_NO)
        while not done:
            state = train_states[train_env.current_step]
            action            = agents[state].select_action(obs)
            obs2, reward, done, _ = train_env.step(action)
            agents[state].memory.push(obs, action, reward, obs2, float(done))
            cnts[state]+=1
            if cnts[state]%64==0:
                agents[state].learn()
            ep_reward += reward
            obs        = obs2

        for i in range(MARKOV_STATES_NO):
            agents[i].decay_epsilon()
        ep_returns.append(ep_reward)

        if ep % 20 == 0:
            val_log = run_episode(val_env, agents, greedy=True)
            val_value = val_log['portfolio_value'].iloc[-1]
            if val_value > best_val_value:
                best_val_value = val_value
                best_state_dicts = [
                    {k: v.clone() for k, v in agents[i].policy_net.state_dict().items()}
                    for i in range(MARKOV_STATES_NO)
                ]

        if ep % 10 == 0:
            recent = np.mean(ep_returns[-10:])
            print(f'Episode {ep:4d}/{N_EPISODES}  '
                f'return={ep_reward:+8.3f}  '
                f'avg10={recent:+8.3f}  '
                f'ε={agents[0].epsilon:.3f}  '
                f'mem={len(agents[0].memory)}')

    print('Training complete.')

    for i in range(MARKOV_STATES_NO):
        agents[i].policy_net.load_state_dict(best_state_dicts[i])

    

    test_env = PairsTradingEnv(
        df_a_test, df_b_test,
        beta            = beta,
        operation_penalty= TRANSACTION_COST,
        initial_cash    = INITIAL_CASH,
        df_a_train = df_a_train,
        df_b_train = df_b_train
    )

    log = run_episode(test_env, agents, greedy=True)

    total_return = log['portfolio_value'].iloc[-1] - INITIAL_CASH
    total_trades = (log['action'] != 1).sum()   # rough: non-HOLD steps
    print(f'Test  total P&L : {total_return:+.4f}')
    print(f'Test  final value: {log["portfolio_value"].iloc[-1]:.4f}')
    print(f'Test  realized PnL: {log["realized_pnl"].iloc[-1]:+.4f}')
    print(f'Test  non-hold steps: {total_trades}')
    returns = log['portfolio_value'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(67800)
    return agents[0], total_return, sharpe_ratio

