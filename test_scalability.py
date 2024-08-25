import numpy as np
import time
import matplotlib.pyplot as plt
from compressed_visibility_matrix import CompressedVisibilityMatrix

def test_scalability(max_size, step_size, sparsity):
    grid_sizes = range(step_size, max_size + 1, step_size)
    compressed_memory = []
    naive_memory = []
    initialization_times = []
    query_times = []

    for size in grid_sizes:
        print(f"Testing grid size: {size}x{size}")
        
        # Generate blocked cells
        num_blocked = int(size * size * sparsity)
        blocked_cells = set()
        while len(blocked_cells) < num_blocked:
            blocked_cells.add((np.random.randint(0, size), np.random.randint(0, size)))

        # Test CompressedVisibilityMatrix
        start_time = time.time()
        cvm = CompressedVisibilityMatrix(size)
        for cell in blocked_cells:
            cvm.add_blocked_cell(*cell)
        initialization_time = time.time() - start_time
        initialization_times.append(initialization_time)

        compressed_memory.append(cvm.get_memory_usage())

        # Measure query time
        num_queries = 1000
        start_time = time.time()
        for _ in range(num_queries):
            from_x, from_y = np.random.randint(0, size), np.random.randint(0, size)
            to_x, to_y = np.random.randint(0, size), np.random.randint(0, size)
            cvm.is_visible(from_x, from_y, to_x, to_y)
        query_time = (time.time() - start_time) / num_queries
        query_times.append(query_time)

        # Calculate naive memory usage (full visibility matrix)
        naive_memory.append((size**4) / (8 * 1024 * 1024))  # in MB

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(grid_sizes, compressed_memory, label='Compressed Visibility Matrix')
    plt.plot(grid_sizes, naive_memory, label='Naive Approach')
    plt.xlabel('Grid Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage vs Grid Size (Sparsity: {sparsity:.2f})')
    plt.legend()
    plt.yscale('log')
    plt.savefig('memory_usage_comparison.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(grid_sizes, initialization_times, label='Initialization Time')
    plt.plot(grid_sizes, query_times, label='Average Query Time')
    plt.xlabel('Grid Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Performance vs Grid Size (Sparsity: {sparsity:.2f})')
    plt.legend()
    plt.yscale('log')
    plt.savefig('performance_comparison.png')
    plt.close()

    # Print results
    print("\nResults:")
    print("Grid Size | Compressed Memory (MB) | Naive Memory (MB) | Init Time (s) | Avg Query Time (s)")
    print("-" * 85)
    for i, size in enumerate(grid_sizes):
        print(f"{size:^9} | {compressed_memory[i]:^22.2f} | {naive_memory[i]:^17.2f} | {initialization_times[i]:^13.4f} | {query_times[i]:^18.6f}")

if __name__ == "__main__":
    test_scalability(max_size=500, step_size=50, sparsity=0.7)