import time
import gc
import numpy as np
import statistics
from buffer import CircularBuffer

# -------------------------------------------------
# Audio Streaming Benchmark
# -------------------------------------------------
def benchmark_audio_buffer(
    capacity: int = 4096,
    frame_size: int = 256,
    iterations: int = 1_000_000,
    trials: int = 7,
    overwriting: bool = False,
    resize: bool = False,
    use_numpy_storage: bool = False,
) -> None:
    """
    Statistically stable benchmark for bulk_enqueue + bulk_dequeue.

    Measures total time per trial instead of per-iteration timing
    to eliminate measurement noise.
    """

    print("Starting Audio Buffer Benchmark...\n")

    # -----------------------------
    # Pre-generate audio data
    # -----------------------------
    input_frame = np.random.randint(
        -32768, 32767, size=frame_size, dtype=np.int16
    )

    warmup_size = capacity // 2
    warmup_data = np.random.randint(
        -32768, 32767, size=warmup_size, dtype=np.int16
    )

    trial_times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()

    try:
        for trial in range(trials):

            # -----------------------------
            # Fresh buffer per trial
            # -----------------------------
            buffer = CircularBuffer(
                capacity=capacity,
                items=None,
                OVERWRITING=overwriting,
                RESIZE=resize,
                NPARR=use_numpy_storage,
                DEBUG=False
            )

            buffer.bulk_enqueue(warmup_data)

            # -----------------------------
            # Warmup loop (CPU/cache stabilize)
            # -----------------------------
            for _ in range(10_000):
                buffer.bulk_enqueue(input_frame)
                buffer.bulk_dequeue(frame_size)

            # -----------------------------
            # Timed section
            # -----------------------------
            start = time.perf_counter()

            for _ in range(iterations):
                buffer.bulk_enqueue(input_frame)
                buffer.bulk_dequeue(frame_size)

            end = time.perf_counter()

            total_time = end - start
            trial_times.append(total_time)

            print(f"Trial {trial+1}: {total_time:.6f} sec")

    finally:
        # Restore GC
        if gc_was_enabled:
            gc.enable()

    # -----------------------------
    # Statistics across trials
    # -----------------------------
    total_samples = iterations * frame_size

    mean_time = statistics.mean(trial_times)
    median_time = statistics.median(trial_times)
    stddev_time = statistics.stdev(trial_times) if len(trial_times) > 1 else 0.0

    samples_per_second = total_samples / median_time
    time_per_sample = median_time / total_samples
    frames_per_second = iterations / median_time

    # -----------------------------
    # Print Results
    # -----------------------------
    print("\n" + "=" * 60)
    print("Audio Buffer Benchmark Results (Stable Mode)")
    print("=" * 60)
    print(f"Capacity:               {capacity}")
    print(f"Frame size:             {frame_size}")
    print(f"Iterations per trial:   {iterations}")
    print(f"Trials:                 {trials}")
    print("-" * 60)
    print(f"Mean total time (s):    {mean_time:.6f}")
    print(f"Median total time (s):  {median_time:.6f}")
    print(f"Std Dev (s):            {stddev_time:.6f}")
    print("-" * 60)
    print(f"Frames / second:        {frames_per_second:,.0f}")
    print(f"Samples / second:       {samples_per_second:,.0f}")
    print(f"Time per sample (s):    {time_per_sample:.12f}")
    print("=" * 60)
    print("\nDone.\n")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    benchmark_audio_buffer(
        capacity=4096,
        frame_size=256,
        iterations=1_000_000,
        trials=5,
        overwriting=False,
        resize=False,
        use_numpy_storage=True
    )
