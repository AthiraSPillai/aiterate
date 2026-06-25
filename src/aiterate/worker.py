from __future__ import annotations

import time

from aiterate.jobs import run_one_optimization_job


def main() -> None:
    print("AIterate worker online. Processing optimizer jobs.")
    while True:
        job = run_one_optimization_job()
        if job:
            print(f"Processed {job.id}: {job.status.value}")
        time.sleep(2)


if __name__ == "__main__":
    main()
