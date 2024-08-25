import numpy as np
import time
import psutil
import os
from memory_profiler import memory_usage
from lossystem import LOSSystem

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MiB

def evaluate_los_system(N, sparsity):
    num_blocked = int(N * N * sparsity)
    blocked_cells = [(np.random.randint(0, N), np.random.randint(0, N)) for _ in range(num_blocked)]
    
    start_time = time.time()
    start_mem = get_process_memory()
    los_system = LOSSystem(N, blocked_cells)
    init_time = time.time() - start_time
    init_memory = get_process_memory() - start_mem
    
    num_updates = min(100, N*N // 10)
    update_cells = [((np.random.randint(0, N), np.random.randint(0, N)), bool(np.random.randint(0, 2))) for _ in range(num_updates)]
    start_time = time.time()
    los_system.update_cells(update_cells)
    update_time = (time.time() - start_time) / num_updates
    
    num_queries = min(1000, N*N)
    query_pairs = [((np.random.randint(0, N), np.random.randint(0, N)), (np.random.randint(0, N), np.random.randint(0, N))) for _ in range(num_queries)]
    start_time = time.time()
    for start, end in query_pairs:
        los_system.is_visible(start, end)
    query_time = (time.time() - start_time) / num_queries
    
    total_memory = get_process_memory()
    
    return {
        "grid_size": N,
        "init_time": init_time,
        "init_memory": init_memory,
        "update_time": update_time,
        "query_time": query_time,
        "total_memory": total_memory
    }

def run_evaluation():
    sparsity = 0.1
    grid_sizes = [10, 20, 50, 100, 200, 500, 1000]
    results = []
    
    for N in grid_sizes:
        print(f"Evaluating grid size: {N}x{N}")
        try:
            result = evaluate_los_system(N, sparsity)
            results.append(result)
            print(f"  Initialization Time: {result['init_time']:.4f} seconds")
            print(f"  Initialization Memory: {result['init_memory']:.2f} MiB")
            print(f"  Total Memory: {result['total_memory']:.2f} MiB")
            print(f"  Average Update Time: {result['update_time']:.6f} seconds")
            print(f"  Average Query Time: {result['query_time']:.6f} seconds")
        except MemoryError:
            print(f"  Memory Error occurred for grid size {N}x{N}")
            break
        except Exception as e:
            print(f"  An error occurred for grid size {N}x{N}: {str(e)}")
            break
        print()
    
    return results

if __name__ == "__main__":
    results = run_evaluation()
    
    print("Summary:")
    print("Grid Size | Init Time (s) | Init Memory (MiB) | Total Memory (MiB) | Avg Update Time (s) | Avg Query Time (s)")
    print("-" * 100)
    for r in results:
        print(f"{r['grid_size']:^9} | {r['init_time']:^13.4f} | {r['init_memory']:^17.2f} | {r['total_memory']:^18.2f} | {r['update_time']:^19.6f} | {r['query_time']:^18.6f}")