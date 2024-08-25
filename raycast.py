import numpy as np
from scipy import sparse

def compute_ray(start, end, N):
    ray = np.zeros(N*N, dtype=bool)
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    for _ in range(n):
        ray[y * N + x] = True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    
    return ray

def generate_ray_matrix(N):
    total_cells = N * N
    rays = []
    for start in range(total_cells):
        for end in range(start + 1, total_cells):
            ray = compute_ray(divmod(start, N), divmod(end, N), N)
            rays.append(ray)
    
    return sparse.csr_matrix(rays)

def compute_initial_visibility(N, blocked_cells):
    R = generate_ray_matrix(N)
    G = sparse.csr_matrix((np.ones(len(blocked_cells)), 
                           ([i for i, j in blocked_cells], [j for i, j in blocked_cells])),
                          shape=(N, N), dtype=bool)
    B = R @ G.reshape(-1, 1)
    V_flat = (B.toarray() == 0).flatten()
    
    return R, V_flat, B

# Example usage
if __name__ == "__main__":
    N = 5
    blocked_cells = [(1, 1), (2, 3)]
    R, V_flat, B = compute_initial_visibility(N, blocked_cells)
    print("Ray matrix shape:", R.shape)
    print("Flat visibility array shape:", V_flat.shape)
    print("Blockage vector shape:", B.shape)
    print("Number of visible cell pairs:", V_flat.sum())