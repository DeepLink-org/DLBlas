import time
import torch


def benchmark_test(fn, fn_triton, args=(), name="gen_fn", times=100, repeat=10):
    print(
        f"--------------------benchmark_{name} for {times * repeat} times--------------------"
    )
    stream = torch.npu.current_stream()
    # warm_up
    stream.synchronize()
    for _ in range(10):
        fn_triton(*args)
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(times * repeat):
        fn_triton(*args)
    stream.synchronize()
    end = time.perf_counter()
    time_compiled = (end - start) / (times * repeat)
    time_compiled *= 1000000
    print(f"time_triton:{time_compiled:.6f}")
    print(f"Runing ref {name} for {times * repeat} times")
    # warm_up
    stream.synchronize()
    for _ in range(10):
        std = fn(*args)
    stream.synchronize()
    start = time.perf_counter()
    for _ in range(times * repeat):
        std = fn(*args)
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    time_eager *= 1000000
    print(f"time_ref:{time_eager:.6f}")
    accelerated = (time_eager - time_compiled) / time_compiled * 100
    print(
        f"Accelerated: {accelerated:.4f}% ref takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us"
    )
    return accelerated, time_eager, time_compiled
