import numpy as np
from collections import defaultdict

class CompressedVisibilityMatrix:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.blocked_cells = set()
        self.visibility_data = defaultdict(dict)

    def add_blocked_cell(self, x, y):
        if (x, y) not in self.blocked_cells:
            self.blocked_cells.add((x, y))
            self._update_visibility(x, y)

    def _update_visibility(self, x, y):
        for dx in range(-self.grid_size + 1, self.grid_size):
            for dy in range(-self.grid_size + 1, self.grid_size):
                if dx == 0 and dy == 0:
                    continue
                self._update_line_of_sight(x, y, dx, dy)

    def _update_line_of_sight(self, x, y, dx, dy):
        line_key = self._get_line_key(dx, dy)
        current_x, current_y = x, y
        distance = 0

        while 0 <= current_x < self.grid_size and 0 <= current_y < self.grid_size:
            if (current_x, current_y) in self.blocked_cells:
                self.visibility_data[line_key][(x, y)] = distance
                return
            current_x += dx
            current_y += dy
            distance += 1

    def _get_line_key(self, dx, dy):
        gcd = np.gcd(dx, dy)
        return (dx // gcd, dy // gcd) if gcd != 0 else (dx, dy)

    def is_visible(self, from_x, from_y, to_x, to_y):
        dx = to_x - from_x
        dy = to_y - from_y
        line_key = self._get_line_key(dx, dy)
        distance = max(abs(dx), abs(dy))

        if line_key in self.visibility_data:
            for (bx, by), block_distance in self.visibility_data[line_key].items():
                if (bx - from_x) * dx >= 0 and (by - from_y) * dy >= 0:
                    relative_distance = max(abs(bx - from_x), abs(by - from_y))
                    if relative_distance <= distance and block_distance <= relative_distance:
                        return False
        return True

    def get_memory_usage(self):
        blocked_cells_memory = len(self.blocked_cells) * 16  # Assuming 8 bytes per coordinate
        visibility_data_memory = sum(
            sum(16 for _ in lines.values())  # 8 bytes for key, 8 for value
            for lines in self.visibility_data.values()
        )
        return (blocked_cells_memory + visibility_data_memory) / (1024 * 1024)  # Convert to MB