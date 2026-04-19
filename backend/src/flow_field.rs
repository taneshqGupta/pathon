use serde::Serialize;
use std::collections::VecDeque;

#[derive(Clone, Serialize)]
pub struct FlowField {
    pub distances: Vec<i32>,
    pub width: i32,
    pub height: i32,
}

impl FlowField {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            distances: vec![i32::MAX; (width * height) as usize],
            width,
            height,
        }
    }

    pub fn update(&mut self, goal: (i32, i32), static_grid: &[u8]) {
        self.distances.fill(i32::MAX);
        let mut queue = VecDeque::new();

        let goal_idx = (goal.1 * self.width + goal.0) as usize;
        self.distances[goal_idx] = 0;
        queue.push_back(goal);

        // 8-Directional offsets
        let directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),   // Orthogonal
            (1, 1), (1, -1), (-1, 1), (-1, -1), // Diagonal
        ];

        while let Some((cx, cy)) = queue.pop_front() {
            let current_idx = (cy * self.width + cx) as usize;
            let current_dist = self.distances[current_idx];

            for (dx, dy) in directions.iter() {
                let nx = cx + dx;
                let ny = cy + dy;

                if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                    let n_idx = (ny * self.width + nx) as usize;
                    
                    // If not a wall and not visited
                    if static_grid[n_idx] == 0 && self.distances[n_idx] == i32::MAX {
                        self.distances[n_idx] = current_dist + 1;
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }
}