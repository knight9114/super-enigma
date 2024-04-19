"""Deep Q-Network Algorithm
"""

# -------------------------------------------------------------------------
#   DQN Runner
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace
from collections import deque
from math import exp, prod
from pathlib import Path
import random
import secrets
import time


# External Library Imports
import gymnasium as gym
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn, optim, Tensor
from torch.utils.tensorboard import SummaryWriter
import tqdm


# -------------------------------------------------------------------------
#   Agent
# -------------------------------------------------------------------------


class Agent(nn.Module):
    def __init__(
        self,
        d_observations: int,
        n_actions: int,
        n_layers: int,
        d_hidden: int,
    ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(d_observations, d_hidden),
            nn.Tanh(),
        )
        self.stack = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden),
                    nn.Tanh(),
                )
                for _ in range(n_layers)
            ]
        )
        self.policy = nn.Linear(d_hidden, n_actions)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.embedding(obs)
        x = self.stack(x)
        return self.policy(x)

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Tensor], metadata: dict[str, str]):
        d_observations = int(metadata["d_observations"])
        n_actions = int(metadata["n_actions"])
        n_layers = int(metadata["n_layers"])
        d_hidden = int(metadata["d_hidden"])

        agent = cls(d_observations, n_actions, n_layers, d_hidden)
        agent.load_state_dict(ckpt)

        return agent


# -------------------------------------------------------------------------
#   Agent Utilities
# -------------------------------------------------------------------------


def load_checkpoint(path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    ckpt = {}
    with safe_open(path, framework="pt", device="cpu") as fp:
        metadata = fp.metadata()
        for key in fp.keys():
            ckpt[key] = fp.get_tensor(key)

    return ckpt, metadata


def save_checkpoint(path: str, agent: Agent):
    ckpt = agent.state_dict()
    metadata = {
        "d_observations": f"{ckpt['embedding.0.weight'].size(1)}",
        "n_actions": f"{ckpt['policy.weight'].size(0)}",
        "n_layers": f"{(len(ckpt) - 4) // 2}",
        "d_hidden": f"{ckpt['policy.weight'].size(1)}",
    }
    save_file(
        tensors=ckpt,
        filename=path,
        metadata=metadata,
    )


# -------------------------------------------------------------------------
#   DQN Utilities
# -------------------------------------------------------------------------


class ExperienceBuffer:
    def __init__(self, capacity: int, ctx: str):
        self.memory = deque([], maxlen=capacity)
        self.ctx = ctx

    def push(self, *args):
        self.memory.append(tuple(args))

    def sample(self, n: int) -> tuple[Tensor, ...]:
        return tuple(
            map(
                lambda lst: torch.from_numpy(np.stack(lst)).to(self.ctx),
                zip(*random.sample(self.memory, n)),
            )
        )

    def __len__(self) -> int:
        return len(self.memory)


def epsilon_greedy_sample_action(qs: Tensor, epsilon: float) -> Tensor:
    if random.random() < epsilon:
        return torch.randint_like(qs[..., 0], qs.size(-1), dtype=torch.long)

    return qs.argmax(-1)


# -------------------------------------------------------------------------
#   Training
# -------------------------------------------------------------------------


def train(
    qnet: Agent,
    env: gym.Env,
    writer: SummaryWriter,
    batch_size: int,
    n_warmup_steps: int,
    n_training_steps: int,
    gamma: float,
    learning_rate: float,
    eps_init: float,
    eps_end: float,
    eps_decay: int,
    experience_buffer_size: int,
    seed: int | None = None,
    ctx: str = "cpu",
):
    seed = secrets.randbits(32) if seed is None else seed
    optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)
    buffer = ExperienceBuffer(experience_buffer_size, ctx)
    seed_offset = 1

    obs, _ = env.reset(seed=seed + seed_offset)
    seed_offset += 1
    rewards = []
    ep_reward = 0
    for t in tqdm.trange(n_warmup_steps + n_training_steps):
        with torch.no_grad():
            epsilon = eps_end + (eps_init - eps_end) * exp(-1.0 * t / eps_decay)
            qvals = qnet(torch.from_numpy(obs.ravel()).to(ctx))
            action = epsilon_greedy_sample_action(qvals, epsilon).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs.ravel(), action, reward, next_obs.ravel(), float(done))
            ep_reward += reward

            obs = next_obs
            if done:
                obs, _ = env.reset(seed=seed + seed_offset)
                seed_offset += 1
                writer.add_scalar("cumulative-rewards", ep_reward, t)
                rewards.append(ep_reward)
                ep_reward = 0

        if t >= n_warmup_steps:
            obs_t, act_t, rew_t, next_t, mask = buffer.sample(batch_size)
            curr_qvals = qnet(obs_t)
            qcurr = curr_qvals.gather(1, act_t[:, None]).squeeze(-1)
            next_qvals = qnet(next_t)
            qnext = next_qvals.gather(1, next_qvals.argmax(-1)[:, None]).squeeze(-1)
            expected_qvals = rew_t + gamma * qnext
            loss = (expected_qvals - qcurr).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss.item(), t)


# -------------------------------------------------------------------------
#   Evaluation
# -------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(agent: Agent, env: gym.Env, seed: int | None = None):
    agent.train(False)
    seed = secrets.randbits(32) if seed is None else seed
    obs, _ = env.reset(seed=seed)
    done = False

    total_reward = 0
    while not done:
        obs_pt = torch.from_numpy(obs.ravel())
        qvals = agent(obs_pt)
        action = qvals.argmax(-1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        env.render()

    print(f"Agent earned a reward of {total_reward}")
    env.close()


# -------------------------------------------------------------------------
#   Bindings
# -------------------------------------------------------------------------


def _train(args: Namespace):
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    env = gym.make(args.env)
    agent = Agent(
        d_observations=prod(env.observation_space.shape),
        n_actions=env.action_space.n,
        n_layers=args.n_layers,
        d_hidden=args.d_hidden,
    ).to(args.ctx)
    writer = SummaryWriter(f"{args.env.lower()}/{timestamp}")
    train(
        qnet=agent,
        env=env,
        writer=writer,
        batch_size=args.batch_size,
        n_warmup_steps=args.n_warmup_steps,
        n_training_steps=args.n_training_steps,
        gamma=args.gamma,
        learning_rate=args.lr,
        eps_init=args.epsilon_start,
        eps_end=args.epsilon_end,
        eps_decay=args.epsilon_decay,
        experience_buffer_size=args.experience_buffer_size,
        seed=args.seed,
        ctx=args.ctx,
    )
    save_checkpoint(f"{args.env.lower()}/{timestamp}/agent.safetensors", agent)


def _evaluate(args: Namespace):
    env = gym.make(args.env, render_mode="human")
    ckpt, metadata = load_checkpoint(args.ckpt)
    agent = Agent.from_checkpoint(ckpt, metadata)
    evaluate(agent, env, args.seed)


# -------------------------------------------------------------------------
#   CLI Utilities
# -------------------------------------------------------------------------


def main(argv: tuple[str] | None = None):
    args = parse_cli_arguments(argv)
    args.execute(args)


def parse_cli_arguments(argv: tuple[str] | None = None) -> Namespace:
    _ENVS: tuple[str] = ("CartPole-v1", "LunarLander-v2")
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    train = subparsers.add_parser("train")
    train.set_defaults(execute=_train)
    agent = train.add_argument_group("Agent Arguments")
    agent.add_argument("--n-layers", type=int, default=1)
    agent.add_argument("--d-hidden", type=int, default=32)
    algo = train.add_argument_group("DQN Arguments")
    algo.add_argument("--env", type=str, choices=_ENVS, default="CartPole-v1")
    algo.add_argument("--batch-size", type=int, default=1024)
    algo.add_argument("--n-warmup-steps", type=int, default=5000)
    algo.add_argument("--n-training-steps", type=int, default=100000)
    algo.add_argument("--gamma", type=float, default=0.99)
    algo.add_argument("--lr", type=float, default=1e-3)
    algo.add_argument("--epsilon-start", type=float, default=1.0)
    algo.add_argument("--epsilon-end", type=float, default=0.01)
    algo.add_argument("--epsilon-decay", type=int, default=500)
    algo.add_argument("--experience-buffer-size", type=int, default=5000)
    algo.add_argument("--ctx", type=str, default="cpu")
    algo.add_argument("--seed", type=int)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.set_defaults(execute=_evaluate)
    evaluate.add_argument("--ckpt", type=Path, required=True)
    evaluate.add_argument("--env", type=str, choices=_ENVS, default="CartPole-v1")
    evaluate.add_argument("--seed", type=int)

    return parser.parse_args(argv)


# -------------------------------------------------------------------------
#   Script Mode
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
