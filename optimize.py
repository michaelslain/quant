"""Shared grid search optimizer with multiprocessing."""

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

    with Pool(ncpu) as pool:
        results = pool.starmap(worker_fn, jobs)

    print(f"  Done. Tested {total} combinations.")
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
