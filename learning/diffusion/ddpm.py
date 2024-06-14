"""Denoising Diffusion Probabilistic Models

This implementation is based on the [paper's code
repository](https://github.com/hojonathanho/diffusion).
"""

# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------

# Standard Library Imports
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from secrets import randbits
from typing import Any, Callable, Iterator

# External Imports
import numpy as np
import jax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray
import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
import optax  # type: ignore
from datasets import Dataset, load_dataset  # type: ignore
from PIL.Image import Resampling
from tensorboardX import SummaryWriter  # type: ignore
import tqdm


# -------------------------------------------------------------------------
#   Setup
# -------------------------------------------------------------------------

flax.config.update("flax_use_orbax_checkpointing", False)


# -------------------------------------------------------------------------
#   Dataset Functions
# -------------------------------------------------------------------------


def create_iterator_factory(
    xs: Array,
    ys: Array | None = None,
    batch_size: int = 128,
) -> Callable[[PRNGKeyArray], Iterator[tuple[Array, Array]]]:
    n = xs.shape[0]
    ys = ys if ys is not None else jnp.zeros((n,))

    def iter_fn(rng) -> Iterator[tuple[Array, Array]]:
        perm = jax.random.permutation(rng, n)

        for j in range(0, n, batch_size):
            yield xs[perm[j : j + batch_size]], ys[perm[j : j + batch_size]]

    return iter_fn


def auto_iterator(
    factory: Callable[[PRNGKeyArray], Iterator[tuple[Array, Array]]],
    rng: PRNGKeyArray,
    n_batches: int,
) -> Iterator[tuple[Array, Array]]:
    idx = 0
    stream = factory(jax.random.fold_in(rng, idx))
    while idx < n_batches:
        try:
            batch = next(stream)
            yield batch
            idx += 1

        except StopIteration:
            stream = factory(jax.random.fold_in(rng, idx))


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
#   Diffusion Functions
# -------------------------------------------------------------------------


def create_betas(start: float, end: float, n: int, rule: str) -> Array:
    match rule:
        case "linear":
            return jnp.linspace(start, end, n)

        case _:
            raise ValueError(f"invalid `{rule=}` provided")


def q_sample(
    x0: Array,
    t: Array,
    noise: Array,
    *,
    sqrt_alpha_bar: Array,
    sqrt_one_minus_alpha_bar: Array,
) -> Array:
    bs, *dims = x0.shape
    ones = (1,) * len(dims)
    a = sqrt_alpha_bar[t].reshape(bs, *ones)
    b = sqrt_one_minus_alpha_bar[t].reshape(bs, *ones)
    return a * x0 + b * noise


# -------------------------------------------------------------------------
#   Training Functions and Utilities
# -------------------------------------------------------------------------


class TrainState(train_state.TrainState):
    rng: PRNGKeyArray


def train(
    net: Unet,
    optimizer: optax.GradientTransformation,
    dataset: Dataset,
    rng: PRNGKeyArray,
    n_train_steps: int,
    writer: SummaryWriter,
    batch_size: int,
    n_diffusion_steps: int,
    betas: Array,
    ema: float,
    keep_n_checkpoints: int,
    checkpoint_every_n_steps: int,
):
    kinit, kdata, kstate, rng = jax.random.split(rng, 4)
    imshape = dataset["image"][0].shape
    vars = net.init(kinit, jnp.empty((1, *imshape)), jnp.empty((1,), dtype=jnp.int32))
    alpha_bar = jnp.cumprod(1 - betas)

    sampler = partial(
        q_sample,
        sqrt_alpha_bar=jnp.sqrt(alpha_bar),
        sqrt_one_minus_alpha_bar=jnp.sqrt(1 - alpha_bar),
    )
    iterator = auto_iterator(
        factory=create_iterator_factory(dataset["image"], dataset["label"], batch_size),
        rng=kdata,
        n_batches=n_train_steps,
    )
    state = TrainState.create(
        apply_fn=net.apply,
        params=vars["params"],
        tx=optimizer,
        rng=kstate,
    )

    for step, (x0, y) in tqdm.tqdm(enumerate(iterator), total=n_train_steps):
        kt, knoise, rng = jax.random.split(rng, 3)
        t = jax.random.randint(kt, (x0.shape[0],), 1, n_diffusion_steps)
        noise = jax.random.normal(knoise, x0.shape)
        xt = sampler(x0, t, noise)

        new_state, metrics = train_step(state=state, xt=xt, t=t, noise=noise)
        state = new_state.replace(
            params=jax.tree_util.tree_map(
                lambda new, old: ema * new + (1 - ema) * old,
                new_state.params,
                state.params,
            )
        )
        writer.add_scalars("training", metrics, step)
        if step % (checkpoint_every_n_steps - 1) == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=Path(writer.logdir) / "checkpoints",
                target=state.params,
                step=step + 1,
                keep=keep_n_checkpoints,
            )


@jax.jit
def train_step(
    state: TrainState,
    xt: Array,
    t: Array,
    noise: Array,
) -> tuple[TrainState, dict[str, Array]]:
    rng = jax.random.fold_in(key=state.rng, data=state.step)
    loss, grads = jax.value_and_grad(
        fun=lambda params: optax.squared_error(
            predictions=state.apply_fn(
                {"params": params},
                xt,
                t,
                train=True,
                rngs={"dropout": rng},
            ),
            targets=noise,
        ).mean()
    )(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics


# -------------------------------------------------------------------------
#   Sampling Functions
# -------------------------------------------------------------------------


def sample():
    pass


# -------------------------------------------------------------------------
#   CLI
# -------------------------------------------------------------------------


def main(argv: tuple[str, ...] | None = None):
    args = parse_cli_arguments(argv)
    args.execute(args)


def parse_cli_arguments(argv: tuple[str, ...] | None = None) -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    trainer = subparsers.add_parser("train")
    trainer.set_defaults(
        execute=lambda args: train(
            net=args.model_factory(args),
            optimizer=args.optimizer_factory(args),
            dataset=args.dataset_factory(args),
            rng=args.rng_factory(args),
            n_train_steps=args.n_train_steps,
            writer=args.writer_factory(args),
            batch_size=args.batch_size,
            n_diffusion_steps=args.n_diffusion_steps,
            betas=args.beta_factory(args),
            ema=args.ema,
            keep_n_checkpoints=args.keep_n_checkpoints,
            checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        ),
        model_factory=lambda args: Unet(
            init_channels=args.d_model,
            blocks_per_multiplier=args.blocks_per_multiplier,
            channel_multipliers=args.multipliers,
            n_groups=args.n_groups,
            dropout=args.dropout,
        ),
        dataset_factory=lambda args: load_dataset(
            args.dataset,
            trust_remote_code=True,
            split="test",
        )
        .map(
            function=partial(
                process_image,
                imkey="image" if args.dataset == "mnist" else "img",
                multiple_of=args.reshape_to_multiple_of,
            ),
            remove_columns="image" if args.dataset == "mnist" else "img",
        )
        .with_format("numpy"),
        optimizer_factory=lambda args: optax.chain(
            optax.clip_by_global_norm(args.clip_grad_norm),
            optax.adamw(
                optax.schedules.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=args.lr,
                    warmup_steps=args.n_warmup_steps,
                    decay_steps=args.n_train_steps,
                ),
            ),
        ),
        writer_factory=lambda args: SummaryWriter(
            logdir=create_experiment_path(
                args.experiment_dir,
                args.experiment_name or args.dataset,
                args.experiment_version,
            ),
            flush_secs=30,
        ),
        rng_factory=lambda args: jax.random.key(args.seed),
        beta_factory=lambda args: create_betas(
            args.beta_start,
            args.beta_end,
            args.n_diffusion_steps,
            args.beta_schedule,
        ),
    )
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
    training.add_argument("--ema", type=float, default=0.9999)

    data = trainer.add_argument_group("Data Hyperparameters")
    data.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    data.add_argument("--reshape-to-multiple-of", type=int, default=32)

    exp = trainer.add_argument_group("Experiment Arguments")
    exp.add_argument("--experiment-dir", type=Path, default=Path("./experiments"))
    exp.add_argument("--experiment-name")
    exp.add_argument("--experiment-version")
    exp.add_argument("--keep-n-checkpoints", type=int, default=5)
    exp.add_argument("--checkpoint-every-n-steps", type=int, default=100)

    sys = trainer.add_argument_group("System Arguments")
    sys.add_argument("--seed", type=int, default=randbits(30))

    sampler = subparsers.add_parser("sample")
    sampler.set_defaults(execute=lambda args: sample())

    return parser.parse_args(argv)


def create_experiment_path(root: Path, name: str, version: str | None = None) -> Path:
    base = root / name
    if version is not None:
        path = base / version
        assert not path.exists()
        return path

    found = list(base.glob("*"))
    idx = 0
    version = "version_{}"
    while version.format(idx) in found:
        idx += 1

    return base / version.format(idx)


def process_image(
    ex: dict[str, Any],
    *,
    imkey: str,
    multiple_of: int,
) -> dict[str, Any]:
    h, w = ex[imkey].size
    c = len(ex[imkey].getbands())
    h = ((h // multiple_of) + 1) * multiple_of
    w = ((w // multiple_of) + 1) * multiple_of

    image = ex[imkey].resize((h, w), Resampling.BILINEAR)
    return {"image": np.asarray(image).reshape([h, w, c])}


# -------------------------------------------------------------------------
#   Script Mode
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()
