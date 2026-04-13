from __future__ import annotations
from typing import List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


# =========================
# LSTM Actor-Critic (same as training)
# =========================
class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 18, hidden: int = 128, n_actions: int = 5):
        super().__init__()
        self.hidden_size = hidden

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden, batch_first=True)

        self.policy = nn.Linear(hidden, n_actions)
        self.value  = nn.Linear(hidden, 1)

    def init_hidden(self, batch: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch, self.hidden_size)
        return z, z.clone()

    def forward(
        self,
        x:  torch.Tensor,
        hc: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        enc = self.encoder(x.reshape(B * T, -1)).reshape(B, T, -1)
        out, hc_new = self.lstm(enc, hc)

        logits = self.policy(out)
        values = self.value(out)

        if squeeze:
            logits = logits.squeeze(1)
            values = values.squeeze(1)

        return logits, values, hc_new


# =========================
# Globals (for efficiency)
# =========================
_model: Optional[LSTMActorCritic] = None
_hc:    Optional[Tuple[torch.Tensor, torch.Tensor]] = None


# =========================
# Load model once
# =========================
def _load_once():
    global _model, _hc

    if _model is not None:
        return

    _model = LSTMActorCritic()

    weight_path = os.path.join(os.path.dirname(__file__), "ppo_diff3_final.pth")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"ppo_diff3_best.pth not found at {weight_path}")

    state = torch.load(weight_path, map_location="cpu")
    _model.load_state_dict(state)

    _model.eval()
    _hc = _model.init_hidden(batch=1)


# =========================
# Policy function
# =========================
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _hc

    _load_once()

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, obs_dim)

    logits, _, _hc = _model(x, _hc)                          # carry hidden state across steps

    best = int(Categorical(logits=logits).mode)               # deterministic argmax

    return ACTIONS[best]


# =========================
# Call this between episodes
# to reset the LSTM memory
# =========================
def reset():
    global _hc
    if _model is not None:
        _hc = _model.init_hidden(batch=1)