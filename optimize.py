"""Shared optimizer with multiprocessing grid search and Bayesian (Optuna) search."""

import time
from multiprocessing import Pool, cpu_count


def grid_search(worker_fn, jobs):
    """
    Run a backtest worker function across all jobs using multiprocessing.

    Args:
        worker_fn: Top-level function (must be picklable) that takes *job args
                   and returns {"sharpe": float, "total_return": float} or None.
        jobs: List of tuples, each tuple is the args for one worker_fn call.

    Returns:
        List of results (same order as jobs).
    """
    ncpu = max(1, cpu_count() // 2)
    total = len(jobs)
    print(f"Testing {total} combinations using {ncpu} cores...")
    t0 = time.time()

    with Pool(ncpu) as pool:
        results = pool.starmap(worker_fn, jobs)

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"  Done. {total} combos in {elapsed:.1f}s ({rate:.0f} combos/s)")
    return results


def find_best(jobs_meta, results):
    """
    Find the best result by Sharpe ratio.

    Args:
        jobs_meta: List of (params_dict, rebalance_every) for each job.
        results: List of metric dicts (or None) from grid_search.

    Returns:
        (best_params, best_rebalance, best_sharpe, best_return) or (None, None, -999, 0)
    """
    best_sharpe = -999
    best_params = None
    best_rebal = None
    best_return = 0

    for (params, rebal), metrics in zip(jobs_meta, results):
        if metrics is None:
            continue
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_params = params
            best_rebal = rebal
            best_return = metrics["total_return"]

    return best_params, best_rebal, best_sharpe, best_return


def bayesian_search(worker_fn, grid, rebalance_options, fixed_args,
                    n_trials=300, n_jobs=None):
    """
    Bayesian optimization using Optuna's TPE sampler.

    Instead of evaluating every grid combination, uses a Tree-structured Parzen
    Estimator to focus on promising parameter regions. Typically finds equivalent
    or better results in 10-20% of the evaluations vs exhaustive grid search.

    Args:
        worker_fn: Top-level function that takes (*fixed_args, kwargs, rebalance_every)
                   and returns {"sharpe": float, "total_return": float} or None.
        grid: Dict of param_name -> list of values to search over.
        rebalance_options: List of rebalance intervals to try.
        fixed_args: Tuple of fixed args to pass before kwargs (e.g., closes_vals, n_cols).
        n_trials: Number of trials to run (default 300).
        n_jobs: Number of parallel workers (default: half CPU cores).

    Returns:
        (best_params, best_rebalance, best_sharpe, best_return) or (None, None, -999, 0)
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if n_jobs is None:
        n_jobs = max(1, cpu_count() // 2)

    # Calculate total grid size for comparison
    total_grid = 1
    for vals in grid.values():
        total_grid *= len(vals)
    total_grid *= len(rebalance_options)

    # If grid is small enough, just do exhaustive search (faster than Optuna overhead)
    if total_grid <= 500:
        print(f"Grid size {total_grid} is small — using exhaustive search.")
        import itertools
        keys = list(grid.keys())
        combos = list(itertools.product(*grid.values()))
        jobs = []
        jobs_meta = []
        for rebal in rebalance_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((*fixed_args, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))
        results = grid_search(worker_fn, jobs)
        return find_best(jobs_meta, results)

    actual_trials = min(n_trials, total_grid)
    print(f"Bayesian search: {actual_trials} trials (vs {total_grid} grid combos, "
          f"{actual_trials/total_grid:.0%} of search space)")
    t0 = time.time()

    best_sharpe = -999
    best_params = None
    best_rebal = None
    best_return = 0
    eval_count = 0

    def objective(trial):
        nonlocal best_sharpe, best_params, best_rebal, best_return, eval_count

        kwargs = {}
        for name, values in grid.items():
            kwargs[name] = trial.suggest_categorical(name, values)

        rebal = trial.suggest_categorical("rebalance_every", rebalance_options)

        result = worker_fn(*fixed_args, kwargs, rebal)
        eval_count += 1

        if result is None:
            return float("-inf")

        sharpe = result["sharpe"]
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = dict(kwargs)
            best_rebal = rebal
            best_return = result["total_return"]

        return sharpe

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    # n_jobs=1: sequential is correct for Bayesian — each trial learns from prior ones
    study.optimize(objective, n_trials=actual_trials, n_jobs=1,
                   show_progress_bar=False)

    elapsed = time.time() - t0
    rate = eval_count / elapsed if elapsed > 0 else 0
    print(f"  Done. {eval_count} evals in {elapsed:.1f}s ({rate:.0f} evals/s)")

    return best_params, best_rebal, best_sharpe, best_return
