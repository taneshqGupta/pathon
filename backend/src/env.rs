use crate::flow_field::FlowField;
use crate::obstacles::ObstacleType;
use pyo3::prelude::*;
use rand::Rng;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

type ModelPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
static MODEL: OnceLock<ModelPlan> = OnceLock::new();

pub fn init_model() {
    if let Ok(model) = tract_onnx::onnx()
        .model_for_path("model.onnx")
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
    {
        MODEL.set(model).unwrap_or_else(|_| ());
        println!("ONNX model successfully loaded via pure Rust (Tract).");
    } else {
        println!("Warning: model.onnx not found. Skipping ONNX integration.");
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Serialize, PartialEq)]
pub enum GameState {
    Setup,
    Playing,
    GameOver,
}

#[pyclass(from_py_object)]
#[derive(Clone, Serialize)]
pub struct GridEnv {
    pub game_state: GameState,
    #[pyo3(get)]
    pub width: i32,
    #[pyo3(get)]
    pub height: i32,
    pub static_grid: Vec<u8>,
    #[pyo3(get)]
    pub agent_pos: (i32, i32),
    #[pyo3(get)]
    pub goal_pos: (i32, i32),
    pub obstacles: Vec<ObstacleType>,
    pub flow_field: FlowField,
    #[pyo3(get)]
    pub fov: Vec<u8>,
    #[pyo3(get, set)]
    pub manual_override: bool,
    #[pyo3(get, set)]
    pub obstacle_density: usize,
    #[pyo3(get, set)]
    pub velocity_multiplier: f32,
    #[pyo3(get)]
    pub goal_reach_count: i32,
    #[pyo3(get)]
    pub collision_count: i32,
    #[pyo3(get)]
    pub replanning_speed_ms: f32,
    #[pyo3(get)]
    pub lidar: Vec<f32>,
    pub fov_history: VecDeque<Vec<u8>>,
    pub lidar_history: VecDeque<Vec<f32>>,
    pub grad_history: VecDeque<(f32, f32)>,
}

#[pymethods]
impl GridEnv {
    #[new]
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            game_state: GameState::Setup,
            width,
            height,
            static_grid: vec![0; (width * height) as usize],
            agent_pos: (10, 10),
            goal_pos: (width - 10, height - 10),
            obstacles: Vec::new(),
            flow_field: FlowField::new(width, height),
            fov: vec![0; 225],
            manual_override: false,
            obstacle_density: 5,
            velocity_multiplier: 1.0,
            goal_reach_count: 0,
            collision_count: 0,
            replanning_speed_ms: 0.0,
            lidar: vec![0.0; 8],
            fov_history: VecDeque::with_capacity(4),
            lidar_history: VecDeque::with_capacity(4),
            grad_history: VecDeque::with_capacity(4),
        }
    }

    pub fn tick(&mut self, input_action: Option<i32>) {
        if self.game_state != GameState::Playing {
            return;
        }

        // Axum Hook: Auto-Infer using ONNX if no manual override
        if input_action.is_none() && !self.manual_override {
            let start_time = std::time::Instant::now();
            let mut chosen_action = 0; // default
            if let Some(plan) = MODEL.get() {
                // Flatten the history buffers
                let mut fov_vec = Vec::with_capacity(4 * 225);
                for frame in &self.fov_history {
                    for &val in frame {
                        fov_vec.push(val as f32);
                    }
                }
                fov_vec.resize(4 * 225, 0.0);
                let mut lidar_vec = Vec::with_capacity(32);
                for l_frame in &self.lidar_history {
                    lidar_vec.extend_from_slice(l_frame);
                }
                lidar_vec.resize(32, 0.0);

                let mut grad_vec = Vec::with_capacity(8);
                for &(gx, gy) in &self.grad_history {
                    grad_vec.push(gx);
                    grad_vec.push(gy);
                }
                grad_vec.resize(8, 0.0);

                let fov_t = tract_ndarray::Array4::from_shape_vec((1, 4, 15, 15), fov_vec)
                    .unwrap()
                    .into_tensor();
                let lidar_t = tract_ndarray::Array2::from_shape_vec((1, 32), lidar_vec)
                    .unwrap()
                    .into_tensor();
                let grad_t = tract_ndarray::Array2::from_shape_vec((1, 8), grad_vec)
                    .unwrap()
                    .into_tensor();

                if let Ok(result) = plan.run(tvec!(fov_t.into(), lidar_t.into(), grad_t.into())) {
                    if let Ok(view) = result[0].to_array_view::<i64>() {
                        if let Some(slice) = view.as_slice() {
                            chosen_action = slice[0] as i32;
                        }
                    }
                }
            }
            let (_, _, _, _, term, _) = self.step_action(chosen_action);
            if term {
                self.game_state = GameState::GameOver;
            }
            self.replanning_speed_ms = start_time.elapsed().as_micros() as f32 / 1000.0;
            return; // We called step_action, which recursively called tick(Some), so the world is updated!
        }

        let start_time = std::time::Instant::now();
        let w = self.width;
        let h = self.height;
        let grid = self.static_grid.clone();

        let vel = self.velocity_multiplier;
        let mut rng = rand::thread_rng();
        let mut ticks_to_run = vel.floor() as i32;
        if rng.gen::<f32>() < (vel - vel.floor()) {
            ticks_to_run += 1;
        }

        for _ in 0..ticks_to_run {
            for obs in &mut self.obstacles {
                obs.update(w, h, &grid);
            }
        }

        let num_obs = self.obstacles.len();
        for i in 0..num_obs {
            for j in (i + 1)..num_obs {
                self.resolve_collision(i, j);
            }
        }

        while self.obstacles.len() < self.obstacle_density {
            self.spawn_random_obstacle();
        }
        while self.obstacles.len() > self.obstacle_density {
            self.obstacles.pop();
        }

        self.fov = self.get_local_fov(15);
        self.lidar = self.get_lidar_readings();

        let (gx, gy) = self.get_flow_gradient(self.agent_pos.0, self.agent_pos.1);
        self.fov_history.push_back(self.fov.clone());
        self.lidar_history.push_back(self.lidar.clone());
        self.grad_history.push_back((gx, gy));

        while self.fov_history.len() > 4 {
            self.fov_history.pop_front();
            self.lidar_history.pop_front();
            self.grad_history.pop_front();
        }

        // replanning_speed_ms ignores the obstacle advancement time since that's engine time
    }

    pub fn reset_env(&mut self) -> (Vec<u8>, Vec<f32>, (f32, f32)) {
        self.reset_with_random_layout();
        self.fov = self.get_local_fov(15);
        self.lidar = self.get_lidar_readings();

        self.fov_history.clear();
        self.lidar_history.clear();
        self.grad_history.clear();

        let (gx, gy) = self.get_flow_gradient(self.agent_pos.0, self.agent_pos.1);

        for _ in 0..4 {
            self.fov_history.push_back(self.fov.clone());
            self.lidar_history.push_back(self.lidar.clone());
            self.grad_history.push_back((gx, gy));
        }

        (self.fov.clone(), self.lidar.clone(), (gx, gy))
    }

    pub fn step_action(&mut self, action: i32) -> (Vec<u8>, Vec<f32>, (f32, f32), f32, bool, bool) {
        let prev_dist = self.get_flow_distance(self.agent_pos.0, self.agent_pos.1);

        let (dx, dy) = match action {
            0 => (0, -1),
            1 => (1, -1),
            2 => (1, 0),
            3 => (1, 1),
            4 => (0, 1),
            5 => (-1, 1),
            6 => (-1, 0),
            7 => (-1, -1),
            _ => (0, 0),
        };

        let next_x = self.agent_pos.0 + dx;
        let next_y = self.agent_pos.1 + dy;

        let mut collided_wall = false;
        let mut can_move = true;
        let agent_radius = 4.0_f32;

        for dx in -4..=4 {
            for dy in -4..=4 {
                let dfx = dx as f32;
                let dfy = dy as f32;
                if dfx * dfx + dfy * dfy <= agent_radius * agent_radius {
                    let cx = next_x + dx;
                    let cy = next_y + dy;
                    if cx < 0
                        || cx >= self.width
                        || cy < 0
                        || cy >= self.height
                        || self.static_grid[(cy * self.width + cx) as usize] == 1
                    {
                        can_move = false;
                        collided_wall = true;
                        break;
                    }
                }
            }
            if !can_move {
                break;
            }
        }

        if can_move {
            self.agent_pos = (next_x, next_y);
        }

        self.tick(Some(action));

        let mut collided_obs = false;
        let ax = self.agent_pos.0;
        let ay = self.agent_pos.1;
        let agent_radius = 4.0_f32;

        let mut agent_cells = Vec::new();
        for dx in -4..=4 {
            for dy in -4..=4 {
                let dfx = dx as f32;
                let dfy = dy as f32;
                if dfx * dfx + dfy * dfy <= agent_radius * agent_radius {
                    agent_cells.push((ax + dx, ay + dy));
                }
            }
        }

        'obs_loop: for obs in &self.obstacles {
            let occupied = obs.get_occupied_cells();
            for cell in &agent_cells {
                if occupied.contains(cell) {
                    collided_obs = true;
                    break 'obs_loop;
                }
            }
        }

        let reached_goal = (self.agent_pos.0 - self.goal_pos.0).abs() <= 1
            && (self.agent_pos.1 - self.goal_pos.1).abs() <= 1;
        let terminated = collided_wall || collided_obs || reached_goal;

        let current_dist = self.get_flow_distance(self.agent_pos.0, self.agent_pos.1);

        let mut reward = if current_dist == 1000.0 {
            -1.0
        } else {
            prev_dist - current_dist
        };

        for l in &self.lidar {
            if *l < 2.0 {
                reward -= 0.02;
            }
        }

        let terminated = collided_wall || reached_goal; // remove collided_obs

        if collided_wall {
            reward -= 5.0;
            self.collision_count += 1;
        } else if collided_obs {
            reward -= 2.0; // penalty but no death
            self.collision_count += 1;
        } else if reached_goal {
            reward += 20.0;
            self.goal_reach_count += 1;
        }

        reward -= 0.005;

        let (gx, gy) = self.get_flow_gradient(self.agent_pos.0, self.agent_pos.1);

        (
            self.fov.clone(),
            self.lidar.clone(),
            (gx, gy),
            reward,
            terminated,
            reached_goal,
        )
    }

    pub fn reset_with_random_layout(&mut self) {
        let mut rng = rand::thread_rng();
        self.obstacles.clear();
        self.static_grid.fill(0);

        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                if x < 4 || x >= w - 4 || y < 4 || y >= h - 4 {
                    self.static_grid[(y * w + x) as usize] = 1;
                }
            }
        }

        let is_maze = false;

        if !is_maze {
            // Big large logical walls
            let num_blocks = rng.gen_range(3..10);
            for _ in 0..num_blocks {
                let bx = rng.gen_range(4..w - 20);
                let by = rng.gen_range(4..h - 20);
                let bw = rng.gen_range(10..w / 3);
                let bh = rng.gen_range(10..h / 3);

                for dy in 0..bh {
                    for dx in 0..bw {
                        if bx + dx < w - 4 && by + dy < h - 4 {
                            self.static_grid[((by + dy) * w + (bx + dx)) as usize] = 1;
                        }
                    }
                }
            }
        } else {
            let cell_size = 20;
            let cols = w / cell_size;
            let rows = h / cell_size;

            for cy in 1..rows {
                for cx in 1..cols {
                    let px = cx * cell_size;
                    let py = cy * cell_size;

                    for dy in 0..4 {
                        for dx in 0..4 {
                            if px + dx < w - 4 && py + dy < h - 4 {
                                self.static_grid[((py + dy) * w + px + dx) as usize] = 1;
                            }
                        }
                    }

                    if rng.gen::<f32>() < 0.45 {
                        if px + 4 < w - 4 && py < h - 4 {
                            let gap_start = rng.gen_range(4..=(cell_size - 8));
                            let gap_end = gap_start + 6; 

                            for dx in 4..cell_size {
                                if dx >= gap_start && dx <= gap_end {
                                    continue;
                                }
                                for dy in 0..4 {
                                    if px + dx < w - 4 && py + dy < h - 4 {
                                        self.static_grid[((py + dy) * w + px + dx) as usize] = 1;
                                    }
                                }
                            }
                        }
                    }

                    if rng.gen::<f32>() < 0.45 {
                        if px < w - 4 && py + 4 < h - 4 {
                            let gap_start = rng.gen_range(4..=(cell_size - 8));
                            let gap_end = gap_start + 6;

                            for dy in 4..cell_size {
                                if dy >= gap_start && dy <= gap_end {
                                    continue;
                                }
                                for dx in 0..4 {
                                    if py + dy < h - 4 && px + dx < w - 4 {
                                        self.static_grid[((py + dy) * w + px + dx) as usize] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.agent_pos = self.find_empty_cell();
        self.goal_pos = self.find_empty_cell();

        let (ax, ay) = self.agent_pos;
        let (gx, gy) = self.goal_pos;
        for dy in -4_i32..=4_i32 {
            for dx in -4_i32..=4_i32 {
                let x1 = (ax + dx).clamp(1, w - 2);
                let y1 = (ay + dy).clamp(1, h - 2);
                let x2 = (gx + dx).clamp(1, w - 2);
                let y2 = (gy + dy).clamp(1, h - 2);
                self.static_grid[(y1 * w + x1) as usize] = 0;
                self.static_grid[(y2 * w + x2) as usize] = 0;
            }
        }

        self.flow_field.update(self.goal_pos, &self.static_grid);

        for _ in 0..5 {
            self.spawn_random_obstacle();
        }
    }

    pub fn get_state_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

impl GridEnv {
    fn spawn_random_obstacle(&mut self) {
        let mut rng = rand::thread_rng();
        let obs_type = rng.gen_range(0..4); // Boulders and 3 varying snakes
        let (x, y) = self.find_empty_cell();

        let obs = match obs_type {
            0 => ObstacleType::Boulder {
                x,
                y,
                vx: rng.gen_range(-1..=1),
                vy: rng.gen_range(-1..=1),
                width: 5,
                height: 5,
            },
            1 => {
                // Long, slow snake
                let length = rng.gen_range(15..30);
                let mut body = Vec::new();
                for i in 0..length {
                    body.push((x - i, y));
                }
                ObstacleType::Snake {
                    body,
                    vx: rng.gen_range(-1..=1),
                    vy: rng.gen_range(-1..=1),
                    thickness: 4,
                    tick_counter: 0,
                    speed_delay: rng.gen_range(4..7), // Slow updates
                }
            }
            2 => {
                // Short, very fast snake
                let length = rng.gen_range(4..10);
                let mut body = Vec::new();
                for i in 0..length {
                    body.push((x - i, y));
                }
                ObstacleType::Snake {
                    body,
                    vx: if rng.gen::<bool>() { 1 } else { -1 },
                    vy: if rng.gen::<bool>() { 1 } else { -1 },
                    thickness: rng.gen_range(2..3),
                    tick_counter: 0,
                    speed_delay: 3, // Fast updates
                }
            }
            _ => {
                // Medium erratic snake
                let length = 8;
                let mut body = Vec::new();
                for i in 0..length {
                    body.push((x - i, y));
                }
                ObstacleType::Snake {
                    body,
                    vx: rng.gen_range(-1..=1),
                    vy: rng.gen_range(-1..=1), // Moving diagonally faster
                    thickness: rng.gen_range(2..3),
                    tick_counter: 0,
                    speed_delay: rng.gen_range(2..4),
                }
            }
        };
        self.obstacles.push(obs);
    }

    fn find_empty_cell(&self) -> (i32, i32) {
        let mut rng = rand::thread_rng();
        loop {
            let x = rng.gen_range(1..self.width - 1);
            let y = rng.gen_range(1..self.height - 1);
            if self.static_grid[(y * self.width + x) as usize] == 0 {
                return (x, y);
            }
        }
    }

    pub fn get_local_fov(&self, size: i32) -> Vec<u8> {
        let mut fov = vec![0; (size * size) as usize];
        let half = size / 2;

        for dy in -half..=half {
            for dx in -half..=half {
                let gx = self.agent_pos.0 + dx;
                let gy = self.agent_pos.1 + dy;
                let fov_idx = ((dy + half) * size + (dx + half)) as usize;

                if gx >= 0 && gx < self.width && gy >= 0 && gy < self.height {
                    if self.static_grid[(gy * self.width + gx) as usize] == 1 {
                        fov[fov_idx] = 1;
                    }
                    for obs in &self.obstacles {
                        if obs.get_occupied_cells().contains(&(gx, gy)) {
                            fov[fov_idx] = 2;
                        }
                    }
                } else {
                    fov[fov_idx] = 1;
                }
            }
        }
        fov
    }

    pub fn get_lidar_readings(&self) -> Vec<f32> {
        let directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ];
        let mut distances = vec![30.0; 8];

        for (i, (dx, dy)) in directions.iter().enumerate() {
            for dist in 1..=30 {
                let tx = self.agent_pos.0 + dx * dist;
                let ty = self.agent_pos.1 + dy * dist;

                if tx < 0
                    || tx >= self.width
                    || ty < 0
                    || ty >= self.height
                    || self.static_grid[(ty * self.width + tx) as usize] == 1
                {
                    distances[i] = dist as f32;
                    break;
                }

                let mut hit_dynamic = false;
                for obs in &self.obstacles {
                    if obs.get_occupied_cells().contains(&(tx, ty)) {
                        hit_dynamic = true;
                        break;
                    }
                }
                if hit_dynamic {
                    distances[i] = dist as f32;
                    break;
                }
            }
        }
        distances
    }

    fn resolve_collision(&mut self, i: usize, j: usize) {
        let (cx1, cy1) = self.get_obs_center(i);
        let (cx2, cy2) = self.get_obs_center(j);

        let dx = cx2 - cx1;
        let dy = cy2 - cy1;
        let dist = (dx * dx + dy * dy) as f32;

        if dist < 400.0 && dist > 0.0 {
            let mut v1u = false;
            let mut v2u = false;

            if let ObstacleType::Boulder {
                ref mut vx,
                ref mut vy,
                ..
            } = self.obstacles[i]
            {
                *vx = if dx < 0 { 1 } else { -1 };
                *vy = if dy < 0 { 1 } else { -1 };
                v1u = true;
            }
            if let ObstacleType::Boulder {
                ref mut vx,
                ref mut vy,
                ..
            } = self.obstacles[j]
            {
                *vx = if dx > 0 { 1 } else { -1 };
                *vy = if dy > 0 { 1 } else { -1 };
                v2u = true;
            }

            if !v1u {
                if let ObstacleType::Snake {
                    ref mut vx,
                    ref mut vy,
                    ..
                } = self.obstacles[i]
                {
                    *vx *= -1;
                    *vy *= -1;
                }
            }
            if !v2u {
                if let ObstacleType::Snake {
                    ref mut vx,
                    ref mut vy,
                    ..
                } = self.obstacles[j]
                {
                    *vx *= -1;
                    *vy *= -1;
                }
            }
        }
    }

    fn get_obs_center(&self, idx: usize) -> (i32, i32) {
        match &self.obstacles[idx] {
            ObstacleType::Boulder {
                x,
                y,
                width,
                height,
                ..
            } => (*x + width / 2, *y + height / 2),
            ObstacleType::Snake { body, .. } => (body[0].0, body[0].1),
        }
    }

    fn get_flow_distance(&self, x: i32, y: i32) -> f32 {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 1000.0;
        }
        let idx = (y * self.width + x) as usize;
        let dist = self.flow_field.distances[idx];
        if dist == i32::MAX {
            1000.0
        } else {
            dist as f32
        }
    }

    fn get_flow_gradient(&self, x: i32, y: i32) -> (f32, f32) {
        let dc = self.get_flow_distance(x, y);
        let dx1 = self.get_flow_distance(x + 1, y);
        let dx2 = self.get_flow_distance(x - 1, y);
        let dy1 = self.get_flow_distance(x, y + 1);
        let dy2 = self.get_flow_distance(x, y - 1);

        let grad_x = if dx1 != 1000.0 && dx2 != 1000.0 {
            (dx1 - dx2) / 2.0
        } else if dx1 != 1000.0 {
            dx1 - dc
        } else if dx2 != 1000.0 {
            dc - dx2
        } else {
            0.0
        };

        let grad_y = if dy1 != 1000.0 && dy2 != 1000.0 {
            (dy1 - dy2) / 2.0
        } else if dy1 != 1000.0 {
            dy1 - dc
        } else if dy2 != 1000.0 {
            dc - dy2
        } else {
            0.0
        };

        (grad_x, grad_y)
    }

    pub fn set_wall(&mut self, x: i32, y: i32, val: u8) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            let idx = (y * self.width + x) as usize;
            if self.static_grid[idx] != val {
                self.static_grid[idx] = val;
                // Recompute flow field after changing grid map
                self.flow_field.update(self.goal_pos, &self.static_grid);
            }
        }
    }

    pub fn toggle_wall(&mut self, x: i32, y: i32) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.static_grid[idx] = 1 - self.static_grid[idx];
            // Recompute flow field after changing grid map
            self.flow_field.update(self.goal_pos, &self.static_grid);
        }
    }

    pub fn move_agent(&mut self, x: i32, y: i32) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            self.agent_pos = (x, y);
            self.fov = self.get_local_fov(15);
            self.lidar = self.get_lidar_readings();
        }
    }

    pub fn move_goal(&mut self, x: i32, y: i32) {
        if x >= 0 && x < self.width && y >= 0 && y < self.height {
            self.goal_pos = (x, y);
            self.flow_field.update(self.goal_pos, &self.static_grid);
        }
    }

    pub fn move_obstacle(&mut self, id: usize, target_x: i32, target_y: i32) {
        if let Some(obs) = self.obstacles.get_mut(id) {
            match obs {
                crate::obstacles::ObstacleType::Snake { body, .. } => {
                    if !body.is_empty() {
                        let dx = target_x - body[0].0;
                        let dy = target_y - body[0].1;
                        for segment in body {
                            segment.0 += dx;
                            segment.1 += dy;
                        }
                    }
                }
                crate::obstacles::ObstacleType::Boulder { x, y, .. } => {
                    *x = target_x;
                    *y = target_y;
                }
            }
            self.fov = self.get_local_fov(15);
            self.lidar = self.get_lidar_readings();
        }
    }
}
