import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from pairs_trading_env import PairsTradingEnv

def dqn(df_a, df_b, SEED=69):
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

    # ── Hyper-parameters ──────────────────────────────────────────────────
    TRANSACTION_COST = 0
    INITIAL_CASH     = 100.0

    GAMMA            = 0.99
    LR               = 1e-3
    BATCH_SIZE       = 64
    MEMORY_SIZE      = 50_000
    TARGET_UPDATE    = 200
    N_EPISODES       = 100

    EPS_START        = 1.00
    EPS_END          = 0.02
    EPS_DECAY        = 0.98
    # ─────────────────────────────────────────────────────────────────────

    STATE_DIM  = PairsTradingEnv.STATE_DIM   # 4  (env observation)
    N_ACTIONS  = PairsTradingEnv.N_ACTIONS   # 3

    # ── CHANGE 1: recurrent memory dimension ─────────────────────────────
    MEMORY_DIM     = 4
    # ── CHANGE 2: derived network I/O sizes ──────────────────────────────
    NET_INPUT_DIM  = STATE_DIM + MEMORY_DIM  # 4 + 4 = 8
    NET_OUTPUT_DIM = N_ACTIONS + MEMORY_DIM  # 3 + 4 = 7
    # ─────────────────────────────────────────────────────────────────────


    # ── Q-Network ─────────────────────────────────────────────────────────
    class QNet(nn.Module):
        # CHANGE 3: added memory_dim param; input is NET_INPUT_DIM=8, output is NET_OUTPUT_DIM=7
        def __init__(self, state_dim: int, n_actions: int, memory_dim: int, hidden: int = 128):
            super().__init__()
            self.n_actions = n_actions
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),   # state_dim = NET_INPUT_DIM = 8
                nn.Linear(hidden, hidden),    nn.ReLU(),
                nn.Linear(hidden, n_actions + memory_dim), # output = NET_OUTPUT_DIM = 7
            )

        # CHANGE 4: forward returns (q_values [3], new_memory [4]) instead of raw tensor
        def forward(self, x: torch.Tensor):
            out = self.net(x)
            return out[:, :self.n_actions], torch.tanh(out[:, self.n_actions:])


    # ── Replay buffer ─────────────────────────────────────────────────────
    # Unchanged — stores 8-dim augmented observations transparently
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
            # CHANGE 5: QNet now receives NET_INPUT_DIM and MEMORY_DIM
            self.policy_net = QNet(NET_INPUT_DIM, N_ACTIONS, MEMORY_DIM).to(DEVICE)
            self.target_net = QNet(NET_INPUT_DIM, N_ACTIONS, MEMORY_DIM).to(DEVICE)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
            self.memory    = ReplayBuffer(MEMORY_SIZE)
            self.steps     = 0
            self.epsilon   = EPS_START
            # CHANGE 6: persistent memory vector carried across steps within an episode
            self.mem       = np.zeros(MEMORY_DIM, dtype=np.float32)

        # CHANGE 7: new method — zero the memory at episode boundaries
        def reset_memory(self) -> None:
            self.mem = np.zeros(MEMORY_DIM, dtype=np.float32)

        # CHANGE 8: select_action now
        #   - concatenates [state (4), self.mem (4)] → 8-dim input
        #   - updates self.mem from the network's memory output
        #   - returns (action, aug_obs) so the caller can push aug_obs into the replay buffer
        def select_action(self, state: np.ndarray):
            aug_obs = np.concatenate([state, self.mem]).astype(np.float32)
            t = torch.tensor(aug_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_vals, new_mem = self.policy_net(t)
                self.mem = new_mem.squeeze(0).cpu().numpy()   # carry memory to next step
            if random.random() < self.epsilon:
                return random.randint(0, N_ACTIONS - 1), aug_obs
            return int(q_vals.argmax(dim=1).item()), aug_obs

        def learn(self):
            if len(self.memory) < BATCH_SIZE:
                return

            s, a, r, s2, done = self.memory.sample(BATCH_SIZE)

            # CHANGE 9: unpack (q_vals, _new_mem) from forward; memory output unused in loss
            q_vals_s, _  = self.policy_net(s)
            q_vals = q_vals_s.gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                # Double-DQN target — same unpacking for policy and target nets
                q_s2, _      = self.policy_net(s2)
                best_actions = q_s2.argmax(dim=1, keepdim=True)
                q_next, _    = self.target_net(s2)
                q_next       = q_next.gather(1, best_actions).squeeze(1)
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


    def make_train_env():
        return PairsTradingEnv(
            df_a_train, df_b_train,
            beta              = beta,
            operation_penalty = TRANSACTION_COST,
            initial_cash      = INITIAL_CASH,
        )

    agent      = DQNAgent()
    train_env  = make_train_env()

    ep_returns = []

    for ep in range(1, N_EPISODES + 1):
        obs  = train_env.reset()
        agent.reset_memory()                        # CHANGE 10: zero memory at episode start
        done = False
        ep_reward = 0.0

        while not done:
            # CHANGE 11: unpack (action, aug_obs); build aug_obs2 with updated memory
            action, aug_obs = agent.select_action(obs)
            obs2, reward, done, _ = train_env.step(action)
            aug_obs2 = np.concatenate([obs2, agent.mem]).astype(np.float32)
            agent.memory.push(aug_obs, action, reward, aug_obs2, float(done))
            agent.learn()
            ep_reward += reward
            obs        = obs2

        agent.decay_epsilon()
        ep_returns.append(ep_reward)

        if ep % 10 == 0:
            recent = np.mean(ep_returns[-10:])
            print(f'Episode {ep:4d}/{N_EPISODES}  '
                f'return={ep_reward:+8.3f}  '
                f'avg10={recent:+8.3f}  '
                f'ε={agent.epsilon:.3f}  '
                f'mem={len(agent.memory)}')

    print('Training complete.')


    def run_episode(env: PairsTradingEnv, agent: DQNAgent, greedy: bool = True):
        """Run one episode; return per-step info dicts."""
        obs  = env.reset(is_test=True)
        agent.reset_memory()            # CHANGE 12: zero memory at episode start
        done = False
        log  = []

        saved_eps = agent.epsilon
        if greedy:
            agent.epsilon = 0.0

        while not done:
            action, _ = agent.select_action(obs)    # CHANGE 13: unpack (action, _aug_obs)
            obs, reward, done, info = env.step(action)
            info['action'] = action
            info['reward'] = reward
            log.append(info)

        agent.epsilon = saved_eps
        return pd.DataFrame(log)


    test_env = PairsTradingEnv(
        df_a_test, df_b_test,
        beta              = beta,
        operation_penalty = TRANSACTION_COST,
        initial_cash      = INITIAL_CASH,
    )

    log = run_episode(test_env, agent, greedy=True)

    total_return = log['portfolio_value'].iloc[-1] - INITIAL_CASH
    total_trades = (log['action'] != 1).sum()
    print(f'Test  total P&L : {total_return:+.4f}')
    print(f'Test  final value: {log["portfolio_value"].iloc[-1]:.4f}')
    print(f'Test  realized PnL: {log["realized_pnl"].iloc[-1]:+.4f}')
    print(f'Test  non-hold steps: {total_trades}')
    return agent, total_return