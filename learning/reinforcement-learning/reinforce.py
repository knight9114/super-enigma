"""REINFORCE Algorithm
"""

# -------------------------------------------------------------------------
#   REINFORCE Runner
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace
from math import prod
from pathlib import Path
import secrets
import time


# External Library Imports
import gymnasium as gym
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn, optim, Tensor
from torch.distributions import Categorical
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
#   REINFORCE Utilities
# -------------------------------------------------------------------------


def sample_action(pi: Tensor) -> tuple[Tensor, Tensor]:
    dist = Categorical(logits=pi)
    actions = dist.sample()
    logprobs = dist.log_prob(actions)
    return actions, logprobs


# -------------------------------------------------------------------------
#   Training
# -------------------------------------------------------------------------


def train(
    agent: Agent,
    env: gym.Env,
    writer: SummaryWriter,
    n_episodes: int,
    gamma: float,
    learning_rate: float,
    seed: int | None = None,
):
    agent.train(True)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    seed = secrets.randbits(32) if seed is None else seed

    for episode in tqdm.trange(n_episodes):
        logprobs, rewards = [], []
        obs, _ = env.reset(seed=seed + episode)
        done = False

        # create trajectory
        while not done:
            obs_pt = torch.from_numpy(obs.ravel())
            pi = agent(obs_pt)
            action, logprob = sample_action(pi)
            obs, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated

            logprobs.append(logprob)
            rewards.append(reward)

        # update agent
        running_g = 0
        tracked_g = []

        for r in rewards[::-1]:
            running_g = r + gamma * running_g
            tracked_g.append(running_g)
        tracked_g = tracked_g[::-1]

        loss = torch.tensor(0.0)
        for logp, g in zip(logprobs, tracked_g):
            loss -= logp.mean() * g

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("cumulative-rewards", sum(rewards), episode)


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
        pi = agent(obs_pt)
        action, logprob = sample_action(pi)
        obs, reward, terminated, truncated, _ = env.step(action.numpy())
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
    )
    writer = SummaryWriter(f"{args.env.lower()}/{timestamp}")
    train(agent, env, writer, args.n_episodes, args.gamma, args.lr, args.seed)
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
    algo = train.add_argument_group("REINFORCE Arguments")
    algo.add_argument("--env", type=str, choices=_ENVS, default="CartPole-v1")
    algo.add_argument("--gamma", type=float, default=0.99)
    algo.add_argument("--n-episodes", type=int, default=5000)
    algo.add_argument("--lr", type=float, default=1e-4)
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
