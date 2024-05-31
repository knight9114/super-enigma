"""Denoising Diffusion Probabilistic Models

This implementation is based on the [paper's code
repository](https://github.com/hojonathanho/diffusion).
"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace
from pathlib import Path

# External Imports
import jax
from jax import numpy as jnp
from jaxtyping import Array
from flax import linen as nn


# -------------------------------------------------------------------------
#   Dataset Functions
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
#   Model Functions
# -------------------------------------------------------------------------


class TimeEmbedding(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, t: Array) -> Array:
        half = self.out_channels // 2
        emb = jnp.log(10000) / (half - 1)
        emb = jnp.exp(jnp.arange(half) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concat([jnp.sin(emb), jnp.cos(emb)], axis=1)
        return emb


class Residual(nn.Module):
    out_channels: int
    n_groups: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: Array, t: Array, *, train: bool = False) -> Array:
        # x.shape = [nbatch, nrows, ncols, nchannels]
        # t.shape = [nbatch, ntime]
        residual = x

        z = nn.GroupNorm(self.n_groups)(x)
        z = nn.swish(z)
        z = nn.Conv(self.out_channels, (3, 3), padding=(1, 1))(z)

        h = nn.swish(t)
        h = nn.Dense(self.out_channels)(h)
        z = z + h[:, None, None, :]

        z = nn.GroupNorm(self.n_groups)(z)
        z = nn.swish(z)
        z = nn.Dropout(self.dropout)(z, deterministic=not train)
        z = nn.Conv(self.out_channels, (3, 3), padding=(1, 1))(z)

        if x.shape[-1] != z.shape[-1]:
            residual = nn.Conv(self.out_channels, (1, 1))(x)

        return z + residual


class Attention(nn.Module):
    out_channels: int
    n_groups: int = 32

    @nn.compact
    def __call__(self, x: Array, t: Array, *, train: bool = False) -> Array:
        # x.shape = [nbatch, nrows, ncols, nchannels]
        # t.shape = [nbatch, ntime]
        b, h, w, c = x.shape

        z = nn.GroupNorm(self.n_groups)(x)
        qkv = nn.Dense(3 * self.out_channels)(z)
        q, k, v = jnp.split(qkv, 3, -1)

        attn = jnp.einsum("bhwc,bHWc->bhwHW", q, k) / jnp.sqrt(c)
        attn = attn.reshape(b, h, w, h * w)
        attn = jax.nn.softmax(attn, -1)
        attn = attn.reshape(b, h, w, h, w)

        scores = jnp.einsum("bhwHW,bHWc->bhwc", attn, v)
        scores = nn.Dense(self.out_channels)(scores)

        return x + scores


class Block(nn.Module):
    out_channels: int
    n_groups: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: Array, t: Array, *, train: bool = False) -> Array:
        z = Residual(
            out_channels=self.out_channels,
            n_groups=self.n_groups,
            dropout=self.dropout,
        )(x, t, train=train)
        z = Attention(
            out_channels=self.out_channels,
            n_groups=self.n_groups,
        )(z, t)
        return z


class PixelShuffle(nn.Module):
    upscale_factor: int = 2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # x.shape = [nbatch, nrows, ncols, nchannels]
        # return.shape = [nbatch, nrows * factor, ncols * factor, nchannels]
        b, h, w, c = x.shape
        z = nn.Conv(x.shape[-1] * self.upscale_factor**2, (3, 3))(x)
        return z.reshape(b, h * self.upscale_factor, w * self.upscale_factor, c)


class PixelUnshuffle(nn.Module):
    downscale_factor: int = 2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # x.shape = [nbatch, nrows, ncols, nchannels]
        # return.shape = [nbatch, nrows // factor, ncols // factor, nchannels]
        b, h, w, c = x.shape
        z = x.reshape(b, h // self.downscale_factor, w // self.downscale_factor, -1)
        return nn.Conv(c, (3, 3))(z)


class Unet(nn.Module):
    init_channels: int = 128
    blocks_per_multiplier: int = 2
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 2)
    n_groups: int = 32
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: Array, t: Array, *, train: bool = False) -> Array:
        # x.shape = [nbatch, nrows, ncols, nchannels]
        # t.shape = [nbatch, ntime]

        # time embedding
        zt = TimeEmbedding(self.init_channels)(t)
        zt = nn.Dense(self.init_channels * 4)(zt)
        zt = nn.swish(zt)
        zt = nn.Dense(self.init_channels * 4)(zt)

        # compression
        zs = [nn.Conv(self.init_channels, (3, 3))(x)]
        for i, mult in enumerate(self.channel_multipliers):
            for _ in range(self.blocks_per_multiplier):
                z = Block(
                    out_channels=self.init_channels * mult,
                    n_groups=self.n_groups,
                    dropout=self.dropout,
                )(zs[-1], zt, train=train)
                zs.append(z)

            if i < len(self.channel_multipliers) - 1:
                zs.append(PixelUnshuffle(2)(zs[-1]))

        # middle
        z = Residual(
            out_channels=self.init_channels * self.channel_multipliers[-1],
            n_groups=self.n_groups,
            dropout=self.dropout,
        )(zs[-1], zt, train=train)
        z = Attention(
            out_channels=self.init_channels * self.channel_multipliers[-1],
            n_groups=self.n_groups,
        )(z, zt, train=train)
        z = Residual(
            out_channels=self.init_channels * self.channel_multipliers[-1],
            n_groups=self.n_groups,
            dropout=self.dropout,
        )(z, zt, train=train)

        # expansion
        for i, mult in enumerate(reversed(self.channel_multipliers)):
            for _ in range(self.blocks_per_multiplier + 1):
                z = Block(
                    out_channels=self.init_channels * mult,
                    n_groups=self.n_groups,
                    dropout=self.dropout,
                )(jnp.concat([z, zs.pop()], axis=-1), zt, train=train)

            if i < len(self.channel_multipliers) - 1:
                z = PixelShuffle(2)(z)

        # projection
        z = nn.GroupNorm(self.n_groups)(z)
        z = nn.swish(z)
        z = nn.Conv(x.shape[-1], (3, 3))(z)

        return z


# -------------------------------------------------------------------------
#   Training Functions
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
#   Sampling Functions
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
#   CLI
# -------------------------------------------------------------------------


def train():
    pass


def sample():
    pass


def main(argv: tuple[str, ...] | None = None):
    args = parse_cli_arguments(argv)
    args.execute(args)


def parse_cli_arguments(argv: tuple[str, ...] | None = None) -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    trainer = subparsers.add_parser("train")
    trainer.set_defaults(execute=lambda args: train())
    model = trainer.add_argument_group("Model Hyperparameters")
    model.add_argument("--d-model", type=int, default=32)
    model.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 2, 2])
    model.add_argument("--blocks-per-multiplier", type=int, default=2)
    model.add_argument("--n-groups", type=int, default=32)
    model.add_argument("--dropout", type=float, default=0.1)

    opt = trainer.add_argument_group("AdamW Hyperparameters")
    opt.add_argument("--lr", type=float, default=2e-4)
    opt.add_argument("--n-warmup-steps", type=int, default=5000)
    opt.add_argument("--clip-grad-norm", type=float, default=1.0)

    diff = trainer.add_argument_group("Diffusion Hyperparameters")
    diff.add_argument("--n-diffusion-steps", type=int, default=1000)
    diff.add_argument("--beta-start", type=float, default=1e-4)
    diff.add_argument("--beta-end", type=float, default=0.02)
    diff.add_argument("--beta-schedule", choices=("linear",), default="linear")

    training = trainer.add_argument_group("Training Hyperparameters")
    training.add_argument("--n-train-steps", type=int, default=100000)
    training.add_argument("--batch-size", type=int, default=128)

    data = trainer.add_argument_group("Data Hyperparameters")
    data.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    data.add_argument("--reshape-to-multiple-of", type=int, default=32)

    exp = trainer.add_argument_group("Experiment Arguments")
    exp.add_argument("--experiment-dir", type=Path, default=Path("./experiments"))
    exp.add_argument("--experiment-name")
    exp.add_argument("--experiment-version")

    sampler = subparsers.add_parser("sample")
    sampler.set_defaults(execute=lambda args: sample())

    return parser.parse_args(argv)


# -------------------------------------------------------------------------
#   Script Mode
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # main()

    net = Unet(64, 1)
    rng = jax.random.key(69)
    kx, kt, knet, rng = jax.random.split(rng, 4)
    x = jax.random.normal(kx, [1, 32, 32, 1])
    t = jax.random.randint(kt, (1,), 0, 100)
    y, _ = net.init_with_output(knet, x, t)

    assert y.shape == x.shape
