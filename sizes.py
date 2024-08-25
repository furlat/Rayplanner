import numpy as np
import matplotlib.pyplot as plt

def size_full_visibility_matrix(N):
    """Size of a full N^2 x N^2 visibility matrix in MB."""
    return (N**4) / (8 * 1024 * 1024)  # Assuming 1 bit per cell

def size_sparse_visibility_matrix(N, sparsity):
    """Size of a sparse visibility matrix in MB."""
    num_elements = int(N**4 * (1 - sparsity))
    return (num_elements * (8 + 4 + 4)) / (1024 * 1024)  # 8 bytes for data, 4 each for row and col indices

def size_ray_to_cells(N):
    """Size of ray-to-cells mapping in MB."""
    num_rays = N**2 * (N**2 - 1) // 2
    avg_cells_per_ray = N  # Approximation
    return (num_rays * avg_cells_per_ray * 4) / (1024 * 1024)  # 4 bytes per cell index

def size_cell_to_rays(N):
    """Size of cell-to-rays mapping in MB."""
    num_cells = N**2
    avg_rays_per_cell = N**2 // 2  # Approximation
    return (num_cells * avg_rays_per_cell * 4) / (1024 * 1024)  # 4 bytes per ray index

def size_blocked_cells_set(N, sparsity):
    """Size of a set of blocked cells in MB."""
    num_blocked = int(N**2 * sparsity)
    return (num_blocked * 8) / (1024 * 1024)  # Assuming 8 bytes per set element

def print_and_plot_memory_usage():
    N_values = np.arange(10, 501, 10)
    sparsities = [0.1, 0.5, 0.9]
    
    print("Memory Usage (MB) for Different Data Structures")
    print("=" * 80)
    print(f"{'N':>5} | {'Full Matrix':>12} | {'Sparse Matrix':>36} | {'Ray-to-Cells':>12} | {'Cell-to-Rays':>12} | {'Blocked Cells Set':>36}")
    print(f"     | {'':>12} | {' '.join([f'Sparsity {s:<9}' for s in sparsities])} | {'':>12} | {'':>12} | {' '.join([f'Sparsity {s:<9}' for s in sparsities])}")
    print("-" * 130)

    plt.figure(figsize=(12, 8))
    
    for N in N_values:
        full_matrix = size_full_visibility_matrix(N)
        sparse_matrices = [size_sparse_visibility_matrix(N, s) for s in sparsities]
        ray_to_cells = size_ray_to_cells(N)
        cell_to_rays = size_cell_to_rays(N)
        blocked_cells = [size_blocked_cells_set(N, s) for s in sparsities]
        
        print(f"{N:5d} | {full_matrix:12.2f} | {' '.join([f'{sm:12.2f}' for sm in sparse_matrices])} | {ray_to_cells:12.2f} | {cell_to_rays:12.2f} | {' '.join([f'{bc:12.2f}' for bc in blocked_cells])}")
        
        if N % 50 == 0:  # Plot every 50th N value to reduce clutter
            plt.plot(N, full_matrix, 'bo', label='Full Matrix' if N == 50 else '')
            for i, s in enumerate(sparsities):
                plt.plot(N, sparse_matrices[i], f'C{i+1}o', label=f'Sparse Matrix (sparsity {s})' if N == 50 else '')
            plt.plot(N, ray_to_cells, 'go', label='Ray-to-Cells Mapping' if N == 50 else '')
            plt.plot(N, cell_to_rays, 'mo', label='Cell-to-Rays Mapping' if N == 50 else '')
            for i, s in enumerate(sparsities):
                plt.plot(N, blocked_cells[i], f'C{i+4}o', label=f'Blocked Cells Set (sparsity {s})' if N == 50 else '')
    
    plt.xlabel('Grid Size (N)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage of Different Data Structures')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('memory_usage_plot.png')
    plt.show()

if __name__ == "__main__":
    print_and_plot_memory_usage()