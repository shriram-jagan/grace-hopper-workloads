import pprint as pp
from collections import defaultdict
from parser import parse_args
from typing import Any, Dict, List, Optional

import cunumeric as np
from legate.core import Machine, Scope, TaskTarget
from legate.core import get_legate_runtime, get_machine
from legate.timing import time


def execute(nelements: np.ndarray, partitions: np.ndarray):

    # Run similar workloads on both CPU and GPU
    def workload(
        x: np.ndarray, y: np.ndarray, start: Optional = 0, end: Optional = None
    ):
        # Mimics a stream triad kernel, x = x*2.0 + y
        np.multiply(x[start:end], 2.0, out=x[start:end])
        np.add(x[start:end], y[start:end], out=x[start:end])

    elapsed_times = defaultdict(dict)
    for elements in nelements:
        runtime.issue_execution_fence(block=True)
        for p in partitions:
            x = np.ones(elements)
            y = 2.0 * np.ones(elements)

            runtime.issue_execution_fence(block=True)
            for rep in range(n_reps):
                if rep == n_skips:
                    start = time("us")
                    runtime.issue_execution_fence(block=True)
                with omps:
                    if p < 1:
                        workload(x, y, start=int(p * elements))

                with gpus:
                    if p > 0:
                        size = int(p * elements)
                workload(x, y, start=0, end=size)

            elapsed_times[int(elements)][round(p, 1)] = ((time("us") - start) / 1e3) / (
                n_reps - n_skips
            )

    return elapsed_times


if __name__ == "__main__":
    args = parse_args()

    partitions = np.array(args.partitions, dtype=float)
    nelements = (1e6 * np.array(args.nelements)).astype(int)

    runtime = get_legate_runtime()
    machine = get_machine()
    omps = machine.only(TaskTarget.OMP)
    gpus = machine.only(TaskTarget.GPU)

    n_omps = machine.count(TaskTarget.OMP)
    n_gpus = machine.count(TaskTarget.GPU)

    n_reps: int = 4
    n_skips: int = 2

    assert n_reps > 0 and n_reps > n_skips
    assert (
        n_omps > 0 and n_gpus > 0
    ), "Number of OpenMP groups and GPUs should be greater than 0"

    print(f"Number of OMPs: {n_omps}")
    print(f"Number of GPUs: {n_gpus}")

    elapsed_times = execute(nelements, partitions)

    print(args, flush=True)
    pp.pp(elapsed_times)
