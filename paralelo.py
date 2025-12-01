import argparse
import numpy as np
import time
import os
from math import sqrt
from mpi4py import MPI
import threading
from typing import List, Tuple


def load_movielens_u_data(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    users = []
    items = []
    raw = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, i, r = int(parts[0]), int(parts[1]), float(parts[2])
            users.append(u); items.append(i); raw.append((u, i, r))
    uniq_users = {u: idx for idx, u in enumerate(sorted(set(users)))}
    uniq_items = {i: idx for idx, i in enumerate(sorted(set(items)))}
    mapped = [(uniq_users[u], uniq_items[i], r) for (u, i, r) in raw]
    return mapped, len(uniq_users), len(uniq_items)

def resize_dataset(ratings, N):
    orig = len(ratings)
    if N is None or N == orig:
        return ratings
    if N < orig:
        return ratings[:N]
    rep = (N // orig) + 1
    extended = ratings * rep
    return extended[:N]

def train_test_split(ratings, test_ratio=0.2, seed=123):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(ratings))
    rng.shuffle(idx)
    split = int(len(ratings) * (1 - test_ratio))
    train_idx = idx[:split]; test_idx = idx[split:]
    train = [ratings[i] for i in train_idx]; test = [ratings[i] for i in test_idx]
    return train, test

def rmse_sum_count(ratings, U, V):
    se = 0.0
    for (i, j, r) in ratings:
        pred = float(U[i].dot(V[j]))
        se += (r - pred) ** 2
    return np.array([se, len(ratings)], dtype=float)

def rmse_from_sum(se, n):
    return sqrt(se / n) if n > 0 else float("nan")


def thread_worker(subbatch: List[Tuple[int,int,float]],
                  U_snapshot: np.ndarray,
                  V_snapshot: np.ndarray,
                  dU: np.ndarray,
                  dV: np.ndarray,
                  lr: float,
                  lambda_reg: float):
    """
    Each thread computes updates into its own dU, dV arrays.
    """
    for (i, j, r) in subbatch:
        # read from snapshots (no race)
        Ui = U_snapshot[i]
        Vj = V_snapshot[j]
        pred = float(Ui.dot(Vj))
        err = r - pred
        # compute delta for U[i] and V[j]
        # note: using += to dU[i]/dV[j] (each thread has its own dU,dV)
        dU[i] += lr * (err * Vj - lambda_reg * Ui)
        dV[j] += lr * (err * Ui - lambda_reg * Vj)


def mf_parallel_threaded_allreduce(ratings,
                                   num_users,
                                   num_items,
                                   k,
                                   lr,
                                   lambda_reg,
                                   epochs,
                                   batch_size,
                                   seed,
                                   num_threads=4,
                                   sequential_time=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # rank 0 prepares shuffle/split and test set
    if rank == 0:
        rng0 = np.random.RandomState(seed)
        idx = np.arange(len(ratings))
        rng0.shuffle(idx)
        shuffled = [ratings[i] for i in idx]
        chunk_size = int(np.ceil(len(shuffled) / size))
        chunks = [shuffled[p*chunk_size : (p+1)*chunk_size] for p in range(size)]
        _, test = train_test_split(ratings, seed=seed)
        meta = (len(ratings), num_users, num_items)
    else:
        chunks = None
        test = None
        meta = None

    # broadcast metadata and test, scatter chunks
    meta = comm.bcast(meta, root=0)
    total_ratings, _, _ = meta
    test = comm.bcast(test, root=0)
    local_train = comm.scatter(chunks, root=0)

    if rank == 0:
        print(f"[MPI-Threaded] Dataset size: {total_ratings}")
        print(f"[MPI-Threaded] Processes: {size}  Threads per process: {num_threads}")
        print(f"[MPI-Threaded] Local chunk size (approx): {len(local_train)}")
        print(f"[MPI-Threaded] Hyperparams: k={k} lr={lr} lambda={lambda_reg} epochs={epochs} batch_size={batch_size}")

    # initialize U, V
    rng = np.random.RandomState(seed + rank)
    U = rng.normal(0, 0.1, (num_users, k))
    V = rng.normal(0, 0.1, (num_items, k))

    total_comm_time = 0.0
    total_start = time.time()

    # epoch loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # go batch by batch (local)
        for bstart in range(0, len(local_train), batch_size):
            batch = local_train[bstart : bstart + batch_size]

            # snapshot for consistent reads by threads
            U_snap = U.copy()
            V_snap = V.copy()

            # prepare per-thread accumulators
            dU_list = [np.zeros_like(U) for _ in range(num_threads)]
            dV_list = [np.zeros_like(V) for _ in range(num_threads)]

            # split batch into num_threads parts
            if len(batch) == 0:
                continue
            chunk_sizes = []
            base = len(batch) // num_threads
            rem = len(batch) % num_threads
            idx0 = 0
            subs = []
            for t in range(num_threads):
                extra = 1 if t < rem else 0
                sz = base + extra
                subs.append(batch[idx0: idx0 + sz])
                idx0 += sz

            threads = []
            for t in range(num_threads):
                if len(subs[t]) == 0:
                    continue
                th = threading.Thread(target=thread_worker,
                                      args=(subs[t], U_snap, V_snap, dU_list[t], dV_list[t], lr, lambda_reg))
                threads.append(th)
                th.start()

            # wait threads
            for th in threads:
                th.join()

            # sum thread-local dU/dV
            dU_total = np.zeros_like(U)
            dV_total = np.zeros_like(V)
            for t in range(num_threads):
                dU_total += dU_list[t]
                dV_total += dV_list[t]

            # apply local update
            U += dU_total
            V += dV_total

            # synchronize across MPI processes: average U and V
            comm_start = time.time()
            U_sum = np.empty_like(U); V_sum = np.empty_like(V)
            comm.Allreduce(U, U_sum, op=MPI.SUM)
            comm.Allreduce(V, V_sum, op=MPI.SUM)
            U = U_sum / size
            V = V_sum / size
            total_comm_time += time.time() - comm_start

        # after epoch compute global RMSE on train (aggregate local parts)
        local_tr = rmse_sum_count(local_train, U, V)
        global_tr = np.zeros(2, float)
        comm.Allreduce(local_tr, global_tr, op=MPI.SUM)
        rmse_train = rmse_from_sum(global_tr[0], global_tr[1])

        # test RMSE computed on broadcast test set; identical on all ranks
        local_te = rmse_sum_count(test, U, V)
        global_te = np.zeros(2, float)
        comm.Allreduce(local_te, global_te, op=MPI.SUM)
        # since each rank had same test and we summed, divide by size
        global_te /= size
        rmse_test = rmse_from_sum(global_te[0], global_te[1])

        if rank == 0:
            epoch_time = time.time() - epoch_start
            print(f"[MPI-Threaded] Epoch {epoch}/{epochs}: RMSE_train={rmse_train:.6f}  RMSE_test={rmse_test:.6f}  (epoch_time={epoch_time:.4f}s)")
            print(f"[MPI-Threaded] Communication time (cum.): {total_comm_time:.6f}s")

    total_time = time.time() - total_start
    if rank == 0:
        print(f"[MPI-Threaded] Total training time: {total_time:.6f}s")
        print(f"[MPI-Threaded] Total communication time: {total_comm_time:.6f}s")
        print(f"[MPI-Threaded] Final RMSE_train={rmse_train:.6f}")
        print(f"[MPI-Threaded] Final RMSE_test={rmse_test:.6f}")
        if sequential_time is not None:
            try:
                speedup = float(sequential_time) / float(total_time)
                efficiency = speedup / float(size)
                print(f"[MPI-Threaded] Speedup vs Sequential (T1={sequential_time:.6f}s): {speedup:.6f}")
                print(f"[MPI-Threaded] Efficiency: {efficiency:.6f}")
            except Exception:
                print("[MPI-Threaded] Could not compute speedup/efficiency (invalid sequential_time)")

    return U, V, total_time, total_comm_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda_reg", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_threads", type=int, default=4, help="Threads per MPI process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequential_time", type=float, default=None)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # load dataset on rank 0
    if rank == 0:
        ratings, num_users, num_items = load_movielens_u_data(args.data_path)
        print(f"[MPI-Threaded] Loaded dataset: {len(ratings)} ratings")
        if args.N is not None:
            ratings = resize_dataset(ratings, args.N)
            print(f"[MPI-Threaded] Resized dataset to N={len(ratings)}")
    else:
        ratings = None
        num_users = None
        num_items = None

    # broadcast metadata and ratings (ratings can be large but this mirrors previous design)
    ratings = comm.bcast(ratings, root=0)
    num_users = comm.bcast(num_users, root=0)
    num_items = comm.bcast(num_items, root=0)

    mf_parallel_threaded_allreduce(ratings=ratings,
                                   num_users=num_users,
                                   num_items=num_items,
                                   k=args.k,
                                   lr=args.lr,
                                   lambda_reg=args.lambda_reg,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   seed=args.seed,
                                   num_threads=args.num_threads,
                                   sequential_time=args.sequential_time)

if __name__ == "__main__":
    main()
