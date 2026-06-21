from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import random

class PairsTradingEnv:
    """Pairs-trading environment for a DQN agent."""
    ACTION_SHORT = 0   # enter / stay short the spread
    ACTION_HOLD  = 1   # do nothing
    ACTION_LONG  = 2   # enter / stay long  the spread

    STATE_DIM  = 7     # length of the observation vector
    N_ACTIONS  = 3     # |action space|

    def __init__(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        beta: float,
        operation_penalty: float = 0.0,
        initial_cash: float = 100.0,
        snapshot_size = 10000,
        interest_free_tick: float = 0.000000558,
        df_a_train: pd.DataFrame = None,
        df_b_train: pd.DataFrame = None,
    ) -> None:
        """
        Parameters
        ----------
        df_a              DataFrame with a 'close' column for stock A.
        df_b              DataFrame with a 'close' column for stock B.
        beta              Hedge ratio (spread = A – beta * B). Constant.
        operation_penalty Penalty per operation (applied only to reward)
        initial_cash      Starting capital (default 100).
        """
        self.prices_a = df_a["close"].values.astype(np.float64)
        self.prices_b = df_b["close"].values.astype(np.float64)
        self.df_a_train = df_a_train
        self.df_b_train = df_b_train

        if len(self.prices_a) != len(self.prices_b):
            raise ValueError("Both price series must have the same length.")

        self.snapshot_size = snapshot_size
        self.beta             = float(beta)
        self.operation_penalty = float(operation_penalty)
        self.initial_cash     = float(initial_cash)
        self.n_steps          = len(self.prices_a)
        self.interest_free_tick = float(interest_free_tick)

        # Pre-compute spread and rolling statistics once
        self.spreads: np.ndarray = self.prices_a - self.beta * self.prices_b
        self._precompute_spread_stats()

        # Mutable episode state (populated by reset)
        self.current_step:   int   = 0
        self.position:       int   = 0      # –1, 0, or +1
        self.n_units:        float = 0.0    # number of spread units held
        self.entry_spread:   float = 0.0    # spread value at last entry
        self.entry_tick: float = 0.0
        self.realized_pnl:   float = 0.0
        self.unrealized_pnl: float = 0.0
        self.min_portfolio_value: float = self.initial_cash
        self.done:           bool  = False


    def mean_std_window(self, window_size):
        bigspread = self.spreads
        if self.df_a_train is not None:
            test_spread = self.df_a_train["close"].values.astype(np.float64) - self.beta * self.df_b_train["close"].values.astype(np.float64)
            bigspread = np.concatenate([test_spread, self.spreads])
        s    = pd.Series(bigspread)
        roll = s.rolling(window=window_size, min_periods=1)
        mean = roll.mean()[-len(self.spreads):]
        std  = roll.std(ddof=1).fillna(1.0).replace(0.0, 1.0)[-len(self.spreads):]
        return mean.values.astype(np.float64), std.values.astype(np.float64)
    
    def _precompute_spread_stats(self) -> None:
        """Compute rolling mean, std, and z-score for the full spread series."""
        self._roll_mean10, self._roll_std10 = self.mean_std_window(10)
        self._roll_mean50, self._roll_std50 = self.mean_std_window(50)
        self._roll_mean100, self._roll_std100 = self.mean_std_window(100)
        self._roll_mean1000, self._roll_std1000 = self.mean_std_window(1000)
        self._roll_mean10000, self._roll_std10000 = self.mean_std_window(10000)
        self._roll_mean40000, self._roll_std40000 = self.mean_std_window(40000)        
    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset(self, is_test = False) -> np.ndarray:
        """
        Reset the environment to the first time step.

        Returns
        -------
        np.ndarray  Initial observation vector.
        """
        self.current_step = 0
        self.last_step = self.n_steps
        if not is_test:
            self.current_step   = random.randint(1, self.n_steps-self.snapshot_size-1)
            self.last_step = self.current_step + self.snapshot_size
        self.position       = 0
        self.n_units        = 0.0
        self.entry_spread   = 0.0
        self.entry_tick = 0.0
        self.realized_pnl   = 0.0
        self.unrealized_pnl = 0.0
        self.min_portfolio_value = self.initial_cash
        self.done           = False
        return self.state()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done — call reset() first.")
        if action not in (0, 1, 2):
            raise ValueError(f"action must be 0, 1, or 2 — got {action}.")

        prev_value = self._portfolio_value()

        pa = self.prices_a[self.current_step]
        pb = self.prices_b[self.current_step]
        s  = self.spreads[self.current_step]

        # ---- Execute trade logic ------------------------------------ #
        if self.position == 0:
            # Flat: opening is the only meaningful action
            if action == self.ACTION_LONG:
                self._open_position(+1, s, pa, pb)
            elif action == self.ACTION_SHORT:
                self._open_position(-1, s, pa, pb)
            else:
                self.realized_pnl += (self.realized_pnl+self.initial_cash)*self.interest_free_tick
            # ACTION_HOLD → interest-free rate

        elif self.position == +1:
            # Long: the only valid close action is ACTION_SHORT
            if action == self.ACTION_SHORT:
                self._close_position(s)
                self._open_position(-1, s, pa, pb)
            elif action == self.ACTION_HOLD:
                self._close_position(s)
            # ACTION_LONG → no-op

        else:  # self.position == -1
            # Short: the only valid close action is ACTION_LONG
            if action == self.ACTION_LONG:
                self._close_position(s)
                self._open_position(+1, s, pa, pb)
            elif action == self.ACTION_HOLD:
                self._close_position(s)
            # ACTION_SHORT → no-op

        # ---- Advance time ------------------------------------------ #
        self.current_step += 1

        # ---- Terminal step ----------------------------------------- #
        if self.current_step >= self.last_step:
            self.done = True
            if self.position != 0:
                # Force-close at the last available price
                self._close_position(self.spreads[self.last_step-1])

        # ---- Update unrealised PnL --------------------------------- #
        self.unrealized_pnl = self._compute_unrealized()
        # ---- Reward = ΔV ------------------------------------------- #
        reward = self._portfolio_value() - prev_value 

        obs = self.state()
        info = {
            "step"            : self.current_step,
            "position"        : self.position,
            "n_units"         : self.n_units,
            "entry_spread"    : self.entry_spread,
            "spread"          : self.spreads[min(self.current_step, self.n_steps - 1)],
            "unrealized_pnl"  : self.unrealized_pnl,
            "realized_pnl"    : self.realized_pnl,
            "portfolio_value" : self._portfolio_value(),
        }
        return obs, reward, self.done, info

    def state(self) -> np.ndarray:
        """
        Return the current observation vector (does NOT advance time).
        """
        idx = min(self.current_step, self.n_steps - 1)

        return np.array(
            [
                float(self.position),
                # (self.spreads[idx]-self._roll_mean10[idx])/self._roll_std10[idx],
                # (self.spreads[idx]-self._roll_mean50[idx])/self._roll_mean50[idx],
                (self.spreads[idx]-self._roll_mean100[idx])/self._roll_std100[idx],
                (self.spreads[idx]-self._roll_mean1000[idx])/self._roll_std1000[idx],
                (self.spreads[idx]-self._roll_mean10000[idx])/self._roll_std10000[idx],
                (self.spreads[idx]-self._roll_mean40000[idx])/self._roll_std40000[idx],
                # self._roll_std1000[idx],
                # self._roll_mean1000[idx],
                # ( self.position * (self.spreads[idx] - self.entry_spread) / (self._roll_std100[idx] + 1e-8)),
                np.log(idx-self.entry_tick+2.0),
                (self.position * (self.spreads[idx] - self.entry_spread)/ (self._roll_std100[idx] + 1e-8)),
                # self._roll_mean100[idx],
                # self._roll_std100[idx],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #

    def valid_actions(self) -> List[int]:
        """
        Return the actions that have a non-trivial effect at the current step.

        The agent may still pass an invalid action (it will be treated as a
        no-op), but masking during training can improve convergence.
        """
        if self.position == 0:
            return [self.ACTION_SHORT, self.ACTION_HOLD, self.ACTION_LONG]
        elif self.position == +1:
            return [self.ACTION_HOLD, self.ACTION_SHORT]   # SHORT closes long
        else:
            return [self.ACTION_HOLD, self.ACTION_LONG]    # LONG  closes short

    @property
    def total_pnl(self) -> float:
        """Realized + unrealized PnL."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def portfolio_value(self) -> float:
        """Current total portfolio value."""
        return self._portfolio_value()

    def __repr__(self) -> str:
        return (
            f"PairsTradingEnv("
            f"step={self.current_step}/{self.n_steps}, "
            f"pos={self.position:+d}, "
            f"upnl={self.unrealized_pnl:+.4f}, "
            f"rpnl={self.realized_pnl:+.4f}, "
            f"value={self._portfolio_value():.4f})"
        )

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _open_position(
        self, direction: int, spread: float, pa: float, pb: float
    ) -> None:
        """Open a long (+1) or short (–1) position at current prices."""
        self.position     = direction
        self.entry_spread = spread
        self.entry_tick = self.current_step
        #TODO: CHANGE THIS SELF.INITIAL_CASH IN HERE AND THE PENALTY
        self.n_units      = self._portfolio_value() / (pa + np.abs(self.beta) * pb)
        self.realized_pnl -= self.operation_penalty*abs(self._portfolio_value())

    def _close_position(self, spread: float) -> None:
        """Close the current open position at the given spread value."""
        gross_pnl          = self.position * self.n_units * (spread - self.entry_spread)
        self.realized_pnl += gross_pnl
        #TODO: CHANGE CHANGE CHANGE
        self.realized_pnl -= self.operation_penalty*abs(self.n_units)*(self.prices_a[self.current_step-1]+abs(self.beta)*self.prices_b[self.current_step-1])    
        self.position      = 0
        self.n_units       = 0.0
        self.entry_spread  = 0.0
        self.entry_tick = self.current_step
        self.unrealized_pnl = 0.0

    def _compute_unrealized(self) -> float:
        """Mark the open position to market."""
        if self.position == 0 or self.current_step >= self.n_steps:
            return 0.0
        return (
            self.position
            * self.n_units
            * (self.spreads[self.current_step] - self.entry_spread)
        )

    def _portfolio_value(self) -> float:
        return self.initial_cash + self.realized_pnl + self.unrealized_pnl