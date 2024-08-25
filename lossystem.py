import numpy as np
from scipy import sparse
from raycast import generate_ray_matrix, compute_initial_visibility

class LOSSystem:
    def __init__(self, N, blocked_cells=None):
        self.N = N
        if blocked_cells is None:
            blocked_cells = []
        self.R, V_flat, self.B = compute_initial_visibility(N, blocked_cells)
        self.V = self._reshape_visibility_matrix(V_flat)
        self.G = sparse.csr_matrix((np.ones(len(blocked_cells)), 
                                    (np.array([i for i, j in blocked_cells]), 
                                     np.array([j for i, j in blocked_cells]))),
                                   shape=(N, N), dtype=int)

    def _reshape_visibility_matrix(self, V_flat):
        V_square = sparse.lil_matrix((self.N**2, self.N**2), dtype=bool)
        ray_index = 0
        for i in range(self.N**2):
            for j in range(i+1, self.N**2):
                if V_flat[ray_index]:
                    V_square[i, j] = V_square[j, i] = True
                ray_index += 1
        for i in range(self.N**2):
            V_square[i, i] = True
        return V_square.tocsr()

    def update_cells(self, cell_changes):
        ΔG = sparse.csr_matrix((np.array([int(new_state) - self.G[i, j] for (i, j), new_state in cell_changes]),
                                (np.array([i for (i, j), _ in cell_changes]),
                                 np.array([j for (i, j), _ in cell_changes]))),
                               shape=(self.N, self.N), dtype=int)
        for (i, j), new_state in cell_changes:
            self.G[i, j] = int(new_state)

        affected_rays = self.R @ ΔG.reshape(-1, 1)
        self.B += affected_rays
        new_V_flat = (self.B.toarray() == 0).flatten()
        self.V = self._reshape_visibility_matrix(new_V_flat)

    def is_visible(self, start, end):
        start_idx = start[0] * self.N + start[1]
        end_idx = end[0] * self.N + end[1]
        return self.V[start_idx, end_idx] != 0

    def get_visible_cells(self, start):
        start_idx = start[0] * self.N + start[1]
        visible = self.V[start_idx].toarray().flatten()
        return [(i // self.N, i % self.N) for i in range(self.N * self.N) if visible[i]]

# Example usage
if __name__ == "__main__":
    N = 10
    blocked_cells = [(2, 2), (3, 3), (4, 4)]
    los_system = LOSSystem(N, blocked_cells)
    
    print("Initial visibility from (0, 0) to (9, 9):", los_system.is_visible((0, 0), (9, 9)))
    
    los_system.update_cells([((5, 5), True)])
    
    print("Visibility after blocking (5, 5):", los_system.is_visible((0, 0), (9, 9)))
    
    visible_cells = los_system.get_visible_cells((0, 0))
    print("Number of cells visible from (0, 0):", len(visible_cells))