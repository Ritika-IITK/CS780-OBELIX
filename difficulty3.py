from __future__ import annotations
import argparse, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
OBS_DIM   = 18
CHUNK_LEN = 64


# ─────────────────────────────────────────────
# Running stats
# ─────────────────────────────────────────────
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x: float):
        self.count += 1
        delta      = x - self.mean
        self.mean += delta / self.count
        self.var  += delta * (x - self.mean)

    @property
    def std(self) -> float:
        return max(float(np.sqrt(self.var / self.count)), 1e-8)

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


# ─────────────────────────────────────────────
# LSTM Actor-Critic  (unchanged from your file)
# ─────────────────────────────────────────────
class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128, n_actions: int = 5):
        super().__init__()
        self.hidden_size = hidden
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU())
        self.lstm    = nn.LSTM(input_size=64, hidden_size=hidden, batch_first=True)
        self.policy  = nn.Linear(hidden, n_actions)
        self.value   = nn.Linear(hidden, 1)

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


# ─────────────────────────────────────────────
# Transition storage
# ─────────────────────────────────────────────
@dataclass
class Step:
    obs:  np.ndarray
    a:    int
    logp: float
    r:    float
    v:    float
    done: bool
    h:    np.ndarray   # float32, shape (1, 1, hidden)
    c:    np.ndarray   # float32, shape (1, 1, hidden)


# ─────────────────────────────────────────────
# GAE
# ─────────────────────────────────────────────
def compute_gae(
    traj:     List[Step],
    last_val: float = 0.0,
    gamma:    float = 0.99,
    lam:      float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    adv, gae = [], 0.0
    vals     = [t.v for t in traj] + [last_val]
    for i in reversed(range(len(traj))):
        mask  = 1.0 - float(traj[i].done)
        delta = traj[i].r + gamma * vals[i + 1] * mask - vals[i]
        gae   = delta + gamma * lam * mask * gae
        adv.insert(0, gae)
    returns = [adv[i] + traj[i].v for i in range(len(traj))]
    return np.array(adv, dtype=np.float32), np.array(returns, dtype=np.float32)


# ─────────────────────────────────────────────
# PPO update with BPTT chunks
# ─────────────────────────────────────────────
def ppo_update(
    net:      LSTMActorCritic,
    opt:      optim.Optimizer,
    traj:     List[Step],
    adv:      np.ndarray,
    ret:      np.ndarray,
    clip:     float = 0.2,
    epochs:   int   = 8,
    ent_coef: float = 0.05,
    chunk_len:int   = CHUNK_LEN,
):
    T       = len(traj)
    obs_arr = np.stack([t.obs  for t in traj])
    act_arr = np.array([t.a    for t in traj])
    lp_arr  = np.array([t.logp for t in traj], dtype=np.float32)
    adv_t   = torch.tensor(adv, dtype=torch.float32)
    ret_t   = torch.tensor(ret, dtype=torch.float32)
    adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(epochs):
        chunk_starts = list(range(0, T, chunk_len))
        random.shuffle(chunk_starts)

        for cs in chunk_starts:
            ce  = min(cs + chunk_len, T)
            idx = slice(cs, ce)

            obs_chunk = torch.tensor(obs_arr[idx], dtype=torch.float32).unsqueeze(0)
            act_chunk = torch.tensor(act_arr[idx])
            lp_chunk  = torch.tensor(lp_arr[idx])
            adv_chunk = adv_t[idx]
            ret_chunk = ret_t[idx]

            # FIX 3 — explicit float32 cast when reconstructing hidden state
            h0 = torch.tensor(traj[cs].h, dtype=torch.float32)
            c0 = torch.tensor(traj[cs].c, dtype=torch.float32)

            logits, values, _ = net(obs_chunk, (h0, c0))
            logits = logits.squeeze(0)
            values = values.squeeze(0).squeeze(-1)

            dist   = Categorical(logits=logits)
            logp   = dist.log_prob(act_chunk)
            ratio  = torch.exp(logp - lp_chunk)

            s1 = ratio * adv_chunk
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_chunk

            policy_loss = -torch.min(s1, s2).mean()
            value_loss  = (ret_chunk - values).pow(2).mean()
            entropy     = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - ent_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def euclidean(ax, ay, bx, by) -> float:
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   required=True)
    ap.add_argument("--episodes",       type=int,   default=4000)
    ap.add_argument("--max_steps",      type=int,   default=1500)
    ap.add_argument("--difficulty",     type=int,   default=3)
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--pretrain",       type=str,   default=None)
    ap.add_argument("--lr",             type=float, default=2e-4)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lam",            type=float, default=0.95)
    ap.add_argument("--clip",           type=float, default=0.2)
    ap.add_argument("--epochs",         type=int,   default=8)
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    OBELIX = import_obelix(args.obelix_py)
    net    = LSTMActorCritic(obs_dim=OBS_DIM)

    if args.pretrain:
        state = torch.load(args.pretrain, map_location="cpu")
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"Loaded {args.pretrain}. Missing: {missing}. Unexpected: {unexpected}")

    opt     = optim.Adam(net.parameters(), lr=args.lr)
    ret_rms = RunningMeanStd()
    best_return = -1e9

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=args.max_steps,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            wall_obstacles=args.wall_obstacles,
        )

        obs        = env.reset()
        hc         = net.init_hidden(batch=1)
        traj:      List[Step] = []
        ep_ret     = 0.0
        prev_dist  = euclidean(env.box_center_x, env.box_center_y,
                                env.bot_center_x, env.bot_center_y)

        # ── Rollout ───────────────────────────────────────────────────
        for _ in range(args.max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits, value, hc_new = net(obs_t, hc)
                dist  = Categorical(logits=logits)
                a     = dist.sample()
                logp  = dist.log_prob(a)

            # FIX 3 — save hidden as explicit float32 arrays
            h_np = hc[0].float().numpy().copy()
            c_np = hc[1].float().numpy().copy()

            obs2, r, done = env.step(ACTIONS[a.item()], render=False)
            hc = hc_new

            # ── Reward shaping (FIX 1 & 5) ───────────────────────────
            # All shaping is applied at raw scale BEFORE normalisation.
            # Values are deliberately large enough to survive RMS compression.

            # 1. Approach bonus — reward closing distance to box
            curr_dist = euclidean(env.box_center_x, env.box_center_y,
                                   env.bot_center_x, env.bot_center_y)
            if not env.enable_push:
                approach_delta = prev_dist - curr_dist
                # FIX 1: scale up so it survives normalisation (was 0.3, now 3.0)
                r += 3.0 * approach_delta
            prev_dist = curr_dist

            # 2. Explicit forward reward — clear signal to move rather than spin
            # FIX 5: was missing; add +2 for FW, small penalty for pure turns
            action_name = ACTIONS[a.item()]
            if action_name == "FW":
                r += 2.0
            else:
                r -= 0.5   # FIX 2: was −0.1 (lost in noise), now −0.5

            # 3. Normalise AFTER all shaping is applied
            ret_rms.update(r)
            r_norm = ret_rms.normalize(r)
            # ─────────────────────────────────────────────────────────

            traj.append(Step(obs, a.item(), logp.item(), r_norm,
                             value.item(), done, h_np, c_np))
            obs     = obs2
            ep_ret += r
            if done:
                break

        # ── Bootstrap ────────────────────────────────────────────────
        last_val = 0.0
        if not traj[-1].done:
            with torch.no_grad():
                obs_t    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, lv, _ = net(obs_t, hc)
                last_val = lv.item()

        adv, ret = compute_gae(traj, last_val=last_val,
                               gamma=args.gamma, lam=args.lam)

        # FIX 4 — slower entropy decay, higher floor
        # Was: 0.05 * 0.997^ep, floor 0.005
        # Now: 0.05 * 0.999^ep, floor 0.01
        ent_coef = max(0.05 * (0.999 ** ep), 0.01)

        ppo_update(net, opt, traj, adv, ret,
                   clip=args.clip, epochs=args.epochs, ent_coef=ent_coef)

        # ── Checkpoint ───────────────────────────────────────────────
        if ep_ret > best_return:
            best_return = ep_ret
            torch.save(net.state_dict(), "ppo_diff3_best.pth")

        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1:4d} | Return: {ep_ret:8.1f} | "
                  f"Best: {best_return:8.1f} | ent_coef: {ent_coef:.4f}")

    torch.save(net.state_dict(), "ppo_diff3_final.pth")
    print("Done. Saved ppo_diff3_final.pth")


if __name__ == "__main__":
    main()