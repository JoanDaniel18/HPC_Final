import argparse
import numpy as np
import time
import os
from math import sqrt
from typing import List, Tuple


def load_movielens_u_data(path: str) -> Tuple[List[Tuple[int, int, float]], int, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    users = []
    items = []
    ratings_raw = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, i, r = int(parts[0]), int(parts[1]), float(parts[2])
            users.append(u)
            items.append(i)
            ratings_raw.append((u, i, r))

    uniq_users = {u: idx for idx, u in enumerate(sorted(set(users)))}
    uniq_items = {i: idx for idx, i in enumerate(sorted(set(items)))}

    ratings = [(uniq_users[u], uniq_items[i], r) for (u, i, r) in ratings_raw]
    return ratings, len(uniq_users), len(uniq_items)


def resize_dataset(ratings: List[Tuple[int, int, float]], N: int) -> List[Tuple[int, int, float]]:
    orig_len = len(ratings)
    if N is None or N == orig_len:
        return ratings

    if N < orig_len:
        return ratings[:N]

    reps = (N // orig_len) + 1
    extended = ratings * reps
    return extended[:N]


def train_test_split(ratings: List[Tuple[int, int, float]], test_ratio=0.2, seed=123):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ratings))
    rng.shuffle(idx)
    split = int(len(ratings) * (1.0 - test_ratio))
    train_idx = idx[:split]
    test_idx = idx[split:]
    train = [ratings[i] for i in train_idx]
    test = [ratings[i] for i in test_idx]
    return train, test


def compute_rmse(ratings: List[Tuple[int, int, float]], U: np.ndarray, V: np.ndarray) -> float:
    se = 0.0
    for (i, j, r) in ratings:
        pred = float(U[i].dot(V[j]))
        se += (r - pred) ** 2
    return sqrt(se / len(ratings)) if len(ratings) > 0 else float("nan")


def mf_sgd_sequential(train: List[Tuple[int, int, float]],
                      test: List[Tuple[int, int, float]],
                      num_users: int,
                      num_items: int,
                      k: int,
                      lr: float,
                      lambda_reg: float,
                      epochs: int,
                      batch_size: int,
                      seed: int):

    rng = np.random.RandomState(seed)
    U = rng.normal(0, 0.1, (num_users, k))
    V = rng.normal(0, 0.1, (num_items, k))

    total_ratings = len(train) + len(test)
    print(f"[Sequential] Dataset size: {total_ratings}")
    print(f"[Sequential] Parameters: k={k} lr={lr} lambda={lambda_reg} epochs={epochs} batch_size={batch_size} seed={seed}")

    start_total = time.time()
    train_arr = np.array(train, dtype=float)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        rng.shuffle(train_arr)
        for bstart in range(0, len(train_arr), batch_size):
            batch = train_arr[bstart:bstart + batch_size]
            for row in batch:
                i = int(row[0]); j = int(row[1]); r = float(row[2])
                pred = U[i].dot(V[j])
                err = r - pred

                Ui = U[i].copy()
                Vj = V[j].copy()

                U[i] += lr * (err * Vj - lambda_reg * Ui)
                V[j] += lr * (err * Ui - lambda_reg * Vj)

        rmse_train = compute_rmse(train, U, V)
        rmse_test = compute_rmse(test, U, V)
        epoch_time = time.time() - epoch_start

        print(f"[Sequential] Epoch {epoch}/{epochs}: RMSE_train={rmse_train:.6f}  RMSE_test={rmse_test:.6f}  (epoch_time={epoch_time:.4f}s)")

    total_time = time.time() - start_total
    print(f"[Sequential] Total training time: {total_time:.6f} seconds")
    print(f"[Sequential] Final RMSE_train={rmse_train:.6f}")
    print(f"[Sequential] Final RMSE_test={rmse_test:.6f}")

    return U, V, total_time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to MovieLens u.data (required)")

    parser.add_argument("--N", type=int, default=None,
                        help="Total number of rating instances to use.")

    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    try:
        ratings, num_users, num_items = load_movielens_u_data(args.data_path)
        print(f"[Sequential] Loaded dataset from: {args.data_path}")
        print(f"[Sequential] Original dataset size: {len(ratings)}")
    except Exception as e:
        print(f"[Sequential] ERROR loading dataset: {e}")
        raise SystemExit(1)

    if args.N is not None:
        ratings = resize_dataset(ratings, args.N)
        print(f"[Sequential] Resized dataset to N={len(ratings)}")

    train, test = train_test_split(ratings, test_ratio=0.2, seed=args.seed)

    mf_sgd_sequential(
        train=train,
        test=test,
        num_users=num_users,
        num_items=num_items,
        k=args.k,
        lr=args.lr,
        lambda_reg=args.lambda_reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
