# Tiny Aristotle: A tiny implementation of the Transformer model for language modeling.

import os
import math
import time
from functools import partial
from typing import Tuple

import numpy as np
from tqdm import tqdm  # For progress bar

import datasets
from boloco import boloco
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from logger import get_logger

logger = get_logger()


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool,
    ):
        super().__init__()
        logger.info(
            f"Initializing TransformerLM with {num_layers} layers, "
            f"{dims} dimensions, and {num_heads} heads."
        )
        logger.info(f"Vocabulary size: {vocab_size}")

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def load_dataset(dataset_name) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_name.startswith("boloco"):
        dataset_class_name = dataset_name.split("_")[1]
        if dataset_class_name is None:
            raise ValueError("Please specify a dataset class name.")
        vocab, train, valid, test = boloco.load_dataset(
            dataset_class_name=dataset_class_name
        )
    else:
        vocab, train, valid, test = datasets.load_dataset(dataset_name)

    return vocab, train, valid, test


def get_model_name(args, train_loss=None):
    """
    Generate a model name based on training parameters.

    Parameters:
        args: An object or namespace containing training configuration (e.g., dataset, batch_size, etc.).
        train_loss: Optional training loss value to include in the filename.

    Returns:
        str: The generated model filename.
    """
    # Core components of the filename
    components = [
        f"{args.dataset}",
        f"b{args.batch_size}",
        f"cs{args.context_size}",
        f"blocks{args.num_blocks}",
        f"dim{args.dim}",
        f"heads{args.num_heads}",
        f"it{args.num_iters}",
        f"lr{args.learning_rate:.0e}",
    ]

    # Include train loss if provided
    if train_loss is not None:
        try:
            components.append(f"loss{train_loss:.0e}")
        except (TypeError, ValueError):
            raise ValueError("train_loss must be a numeric value if provided.")

    # Join components and add file extension
    model_name = "_".join(components) + ".npz"
    return model_name


def main(args):
    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report

    vocab, train, valid, test = load_dataset(args.dataset)

    # Initialize model:
    model = TransformerLM(
        len(vocab), args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )
    model_name = get_model_name(args)
    logger = get_logger(log_file=f"logs/{model_name}.log")
    mx.eval(model.parameters())
    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(
        f"Training transformer model '{model_name}' with {nparams / 1024**2:.3f} M parameters"
    )

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )

    def eval_fn(dataset):
        if dataset.size == 0:
            return float("inf")
        inputs, targets = map(mx.array, to_samples(context_size, dataset[:10_000]))
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
            bx, by = map(mx.array, (bx, by))
            losses = loss_fn(model, bx, by, reduce=False)
            loss += mx.sum(losses).item()
        return loss / len(targets)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []

    tic = time.perf_counter()

    for it, (inputs, targets) in zip(range(args.num_iters), train_iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        optimizer.learning_rate = min(1, it / args.lr_warmup) * args.learning_rate
        loss = step(inputs, targets)
        mx.eval(state)
        losses.append(loss.item())

        # Update progress bar with ETA and loss
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            logger.info(
                f"model={model_name} "
                f"iter={it + 1}/{args.num_iters} "
                f"train_loss={train_loss:.3f} "
                f"train_ppl={math.exp(train_loss):.3f} "
                f"train_timing={(toc - tic):.3f}s "
            )
            losses = []
            tic = time.perf_counter()

        if (it + 1) % steps_per_eval == 0:
            val_loss = eval_fn(valid)
            toc = time.perf_counter()
            logger.info(
                f"model={model_name} "
                f"iter={it + 1}/{args.num_iters} "
                f"val_loss={val_loss:.3f} "
                f"val_ppl={math.exp(train_loss):.3f} "
                f"val_timing={(toc - tic):.3f}s "
            )
            tic = time.perf_counter()

    if args.eval_test:
        test_loss = eval_fn(test)
        logger.info(
            f"model={model_name} "
            f"test_loss={test_loss:.3f} "
            f"test_ppl={math.exp(test_loss):.3f} "
        )

    model_directory = os.path.join("models", f"{args.dataset}")
    os.makedirs(model_directory, exist_ok=True)
    model_full_path = os.path.join(model_directory, model_name)
    print(f"Saving model as {model_full_path}")
    model.save_weights(model_full_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with MLX.")
    parser.add_argument(
        "--gpu", action="store_true", default=True, help="Use the GPU back-end."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="boloco_mt10",
        choices=[
            "enwik8",
            "ptb",
            "wikitext2",
            "wikitext103",
            "boloco_mt5",
            "boloco_mt6",
            "boloco_mt7",
            "boloco_mt8",
            "boloco_mt9",
            "boloco_mt10",
            "boloco_mt11",
            "boloco_mt12",
            "boloco_mt13",
            "boloco_mt14",
            "boloco_mt15",
            "boloco_mt16",
            "boloco_mt17",
            "boloco_mt18",
            "boloco_mt19",
            "boloco_mt20",
        ],
        help="Dataset to train and evaluate on.",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=64,
        help="Context size in tokens of the model.",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=4,
        help="Number of Transformer blocks.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Dimensionality of embeddings and hidden layers.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of heads used for multi-head attention.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Perform gradient checkpointing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10_000,
        help="Iterations to train for.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Set the weight decay.",
    )
    parser.add_argument(
        "--lr_warmup",
        type=int,
        default=100,
        help="LR linear warmup iterations.",
    )
    parser.add_argument(
        "--steps_per_report",
        type=int,
        default=50,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=1_000,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        default=True,
        help="Evaluate on the test set after training.",
    )
    args = parser.parse_args()
    main(args)
