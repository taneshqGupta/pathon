mod env;
mod flow_field;
mod obstacles;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use env::GridEnv;
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tokio::time::{interval, Duration}; // Needed to split socket

struct AppState {
    env: Arc<Mutex<GridEnv>>,
    tx: broadcast::Sender<String>,
}

#[tokio::main]
async fn main() {
    let mut initial_env = GridEnv::new(300, 200);
    initial_env.reset_env();

    let env_state = Arc::new(Mutex::new(initial_env));
    let (tx, _rx) = broadcast::channel(100);

    let app_state = Arc::new(AppState {
        env: env_state.clone(),
        tx: tx.clone(),
    });

    let sim_env = env_state.clone();
    let sim_tx = tx.clone();
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(33));
        loop {
            ticker.tick().await;
            let mut e = sim_env.lock().await;
            e.tick(None);
            if let Ok(json_state) = serde_json::to_string(&*e) {
                let _ = sim_tx.send(json_state);
            }
        }
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("Simulation server running on ws://127.0.0.1:3000/ws");
    env::init_model();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // Task to send simulation state TO the frontend
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Task to receive commands FROM the frontend
    let env_state = state.env.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(Message::Text(text))) = receiver.next().await {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(cmd) = json.get("command") {
                    let mut e = env_state.lock().await;
                    if cmd == "reset" {
                        e.reset_env();
                        e.game_state = env::GameState::Setup;
                    } else if cmd == "start" {
                        e.game_state = env::GameState::Playing;
                    } else if cmd == "toggle_wall" {
                        if let (Some(x), Some(y)) = (
                            json.get("x").and_then(|v| v.as_i64()),
                            json.get("y").and_then(|v| v.as_i64())
                        ) {
                            e.toggle_wall(x as i32, y as i32);
                        }
                    } else if cmd == "set_wall" {
                        if let (Some(x), Some(y), Some(val)) = (
                            json.get("x").and_then(|v| v.as_i64()),
                            json.get("y").and_then(|v| v.as_i64()),
                            json.get("val").and_then(|v| v.as_u64())
                        ) {
                            e.set_wall(x as i32, y as i32, val as u8);
                        }
                    } else if cmd == "move_agent" {
                        if let (Some(x), Some(y)) = (
                            json.get("x").and_then(|v| v.as_i64()),
                            json.get("y").and_then(|v| v.as_i64())
                        ) {
                            e.move_agent(x as i32, y as i32);
                        }
                    } else if cmd == "move_goal" {
                        if let (Some(x), Some(y)) = (
                            json.get("x").and_then(|v| v.as_i64()),
                            json.get("y").and_then(|v| v.as_i64())
                        ) {
                            e.move_goal(x as i32, y as i32);
                        }
                    } else if cmd == "move_obstacle" {
                        if let (Some(id), Some(x), Some(y)) = (
                            json.get("id").and_then(|v| v.as_u64()),
                            json.get("x").and_then(|v| v.as_i64()),
                            json.get("y").and_then(|v| v.as_i64())
                        ) {
                            e.move_obstacle(id as usize, x as i32, y as i32);
                        }
                    } else if cmd == "update_params" {
                        if let Some(den) = json.get("obstacle_density").and_then(|v| v.as_u64()) {
                            e.obstacle_density = den as usize;
                        }
                        if let Some(vel) = json.get("velocity_multiplier").and_then(|v| v.as_f64()) {
                            e.velocity_multiplier = vel as f32;
                        }
                        if let Some(man) = json.get("manual_override").and_then(|v| v.as_bool()) {
                            e.manual_override = man;
                        }
                    } else if cmd == "manual_action" {
                        if let Some(act) = json.get("action").and_then(|v| v.as_i64()) {
                            let (_, _, _, _, term, _) = e.step_action(act as i32);
                            if term {
                                e.game_state = env::GameState::GameOver;
                            }
                        }
                    }
                }
            }
        }
    });

    // If either task fails (client disconnects), abort the other
    tokio::select! {
        _ = (&mut send_task) => recv_task.abort(),
        _ = (&mut recv_task) => send_task.abort(),
    };
}
