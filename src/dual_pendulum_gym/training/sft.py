"""Stage 1: Supervised Fine-Tuning (Behavioral Cloning) from human demonstrations."""
import argparse
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dual_pendulum_gym.training.actor_critic import ActorCritic


def parse_args():
    p = argparse.ArgumentParser(description="SFT: Learn from human demos")
    p.add_argument("--demos", type=str, required=True,
                   help="Path to demo file(s). Glob patterns OK (e.g. 'demos/*.npz')")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--save-path", type=str, default="checkpoints/model_sft.pt",
                   help="Where to save the SFT model")
    p.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    return p.parse_args()


def load_demos(pattern):
    """Load and concatenate all demo files matching the pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No demo files found matching: {pattern}")

    all_obs = []
    all_actions = []
    for f in files:
        data = np.load(f)
        all_obs.append(data["observations"])
        all_actions.append(data["actions"])
        print(f"  Loaded {f}: {len(data['observations'])} steps")

    obs = np.concatenate(all_obs)
    actions = np.concatenate(all_actions)
    print(f"  Total: {len(obs)} steps from {len(files)} file(s)")
    return obs, actions


def train_sft():
    args = parse_args()

    print("Loading demos...")
    obs, actions = load_demos(args.demos)

    # Filter out no-op actions if they dominate (optional rebalancing)
    action_counts = np.bincount(actions, minlength=3)
    print(f"  Action distribution: left={action_counts[0]}, noop={action_counts[1]}, right={action_counts[2]}")

    # Convert to tensors
    obs_t = torch.FloatTensor(obs)
    act_t = torch.LongTensor(actions)

    # Train/val split
    n = len(obs_t)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val
    indices = torch.randperm(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = TensorDataset(obs_t[train_idx], act_t[train_idx])
    val_ds = TensorDataset(obs_t[val_idx], act_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"  Train: {n_train}, Val: {n_val}")

    # Model
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining SFT for {args.epochs} epochs...")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_obs, batch_act in train_loader:
            logits, _ = model(batch_obs)
            loss = criterion(logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_obs)
            train_correct += (logits.argmax(1) == batch_act).sum().item()
            train_total += len(batch_obs)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_obs, batch_act in val_loader:
                logits, _ = model(batch_obs)
                val_correct += (logits.argmax(1) == batch_act).sum().item()
                val_total += len(batch_obs)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3d}/{args.epochs} | "
                f"Loss: {train_loss / train_total:.4f} | "
                f"Train Acc: {train_acc:.1f}% | "
                f"Val Acc: {val_acc:.1f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)

    print(f"\nSFT complete! Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {args.save_path}")
    print(f"\nNext: run RL fine-tuning with:")
    print(f"  python -m dual_pendulum_gym.training.train --pretrained {args.save_path} --render")


if __name__ == "__main__":
    train_sft()
