import random
from typing import Callable
from buffer import CircularBuffer
import statistics
import time


def benchmark(func: Callable, iterations: int, *args, **kwargs) -> tuple[float, float, list[float]]:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    total_time = sum(times)
    avg_time = total_time / len(times)
    return total_time, avg_time, times


def benchmark_circular_queue(capacity: int, iterations: int, overwriting: bool = False, resize: bool = False) -> None:
    LOW_INT : int = -32_768
    HIGH_INT : int = 32_767
    data = [random.randint(LOW_INT, HIGH_INT) for _ in range(capacity)]
    queue = CircularBuffer(
        capacity=capacity,
        items=data,
        OVERWRITING=overwriting,
        RESIZE=resize,
        DEBUG=False
    )

    # Name, iterations, total, avg, min, max, stddev
    results = []

    def record(name: str, total: float, avg: float, times: list[float], iters: int):
        results.append((
            name,
            iters,
            total,
            avg,
            min(times),
            max(times),
            statistics.stdev(times) if len(times) > 1 else 0.0
        ))

    # enqueue
    total, avg, times = benchmark(
    lambda: queue.enqueue(random.randint(LOW_INT, HIGH_INT)),
        iterations
    )
    record("enqueue", total, avg, times, iterations)

    # dequeue 
    total, avg, times = benchmark(
        lambda: (queue.dequeue(), queue.enqueue(random.randint(LOW_INT, HIGH_INT))),
        iterations
    )
    record("dequeue", total, avg, times, iterations)

    # peek
    total, avg, times = benchmark(
        lambda: queue.peek(),
        iterations
    )
    record("peek", total, avg, times, iterations)

    # bulk enqueue
    batch = [random.randint(LOW_INT, HIGH_INT) for _ in range(capacity)]
    total, avg, times = benchmark(
        lambda: queue.bulk_enqueue(batch),
        iterations
    )
    record("bulk_enqueue", total, avg, times, iterations)

    # bulk dequeue
    total, avg, times = benchmark(
        lambda: (queue.bulk_dequeue(capacity), queue.bulk_enqueue(batch)),
        iterations
    )
    record("bulk_dequeue", total, avg, times, iterations)

    # resize (only meaningful if resize enabled)
    if resize:
        total, avg, times = benchmark(
            lambda: queue.resize(),
            iterations
        )
        record("resize", total, avg, times, iterations)

    # build queue
    total, avg, times = benchmark(
        lambda: CircularBuffer(
            capacity=capacity,
            items=data,
            OVERWRITING=overwriting,
            RESIZE=resize
        ),
        iterations
    )
    record("build_buffer", total, avg, times, iterations)

    print_table(results)


def print_table(results) -> None:
    print(f"{'Operation':<25} {'Iterations':>10} {'Total(s)':>10} {'Avg(s)':>10} {'Min(s)':>10} {'Max(s)':>10} {'StdDev':>10}")
    print("-" * 90)

    for op, count, total, avg, mn, mx, stdev in results:
        print(
            f"{op:<25} "
            f"{count:>10} "
            f"{total:>10.4f} "
            f"{avg:>10.8f} "
            f"{mn:>10.8f} "
            f"{mx:>10.8f} "
            f"{stdev:>10.8f}"
        )

if __name__ == "__main__":
    print("Starting Benchmark...")
    benchmark_circular_queue(1_000_000,100,True,False)