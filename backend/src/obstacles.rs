use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub enum ObstacleType {
    Boulder { x: i32, y: i32, vx: i32, vy: i32, width: i32, height: i32 },
    Snake { body: Vec<(i32, i32)>, vx: i32, vy: i32, thickness: i32, tick_counter: i32, speed_delay: i32 },
}

// Helper to check if any part of a bounding box hits a static wall
fn hits_wall(x: i32, y: i32, w: i32, h: i32, gw: i32, gh: i32, grid: &[u8]) -> bool {
    for i in 0..w {
        for j in 0..h {
            let cx = x + i;
            let cy = y + j;
            if cx < 0 || cx >= gw || cy < 0 || cy >= gh { return true; }
            if grid[(cy * gw + cx) as usize] == 1 { return true; }
        }
    }
    false
}

impl ObstacleType {
    pub fn update(&mut self, grid_width: i32, grid_height: i32, static_grid: &[u8]) {
        match self {
            ObstacleType::Boulder { x, y, vx, vy, width, height } => {
                let nx = *x + *vx;
                let ny = *y + *vy;
                
                let mut hit_x = false;
                let mut hit_y = false;

                if hits_wall(nx, *y, *width, *height, grid_width, grid_height, static_grid) {
                    *vx *= -1;
                    hit_x = true;
                }
                if hits_wall(*x, ny, *width, *height, grid_width, grid_height, static_grid) {
                    *vy *= -1;
                    hit_y = true;
                }

                if !hit_x { *x = nx; }
                if !hit_y { *y = ny; }
            }
            ObstacleType::Snake { body, vx, vy, thickness, tick_counter, speed_delay } => {
                *tick_counter += 1;
                if *tick_counter >= *speed_delay {
                    *tick_counter = 0;
                    let head = body[0];
                    let mut nx = head.0 + *vx;
                    let mut ny = head.1 + *vy;
                    
                    let th = *thickness;
                    if hits_wall(nx, head.1, th, th, grid_width, grid_height, static_grid) {
                        *vx *= -1;
                        nx = head.0 + *vx;
                    }
                    if hits_wall(head.0, ny, th, th, grid_width, grid_height, static_grid) {
                        *vy *= -1;
                        ny = head.1 + *vy;
                    }

                    body.insert(0, (nx, ny));
                    body.pop();
                }
            }
        }
    }

    pub fn get_occupied_cells(&self) -> Vec<(i32, i32)> {
        let mut cells = Vec::new();
        match self {
            ObstacleType::Boulder { x, y, width, height, .. } => {
                let w_f32 = *width as f32;
                let h_f32 = *height as f32;
                let center_x = *x as f32 + w_f32 / 2.0;
                let center_y = *y as f32 + h_f32 / 2.0;
                let radius = w_f32.max(h_f32) / 2.0; // Fit the circle radius
                
                for i in 0..*width {
                    for j in 0..*height {
                        let cell_x = *x + i;
                        let cell_y = *y + j;
                        
                        let dx = (cell_x as f32 + 0.5) - center_x;
                        let dy = (cell_y as f32 + 0.5) - center_y;
                        if dx * dx + dy * dy <= radius * radius {
                            cells.push((cell_x, cell_y));
                        }
                    }
                }
            }
            ObstacleType::Snake { body, thickness, .. } => {
                for &(bx, by) in body {
                    for i in 0..*thickness {
                        for j in 0..*thickness {
                            cells.push((bx + i, by + j));
                        }
                    }
                }
            }
        }
        cells
    }
}