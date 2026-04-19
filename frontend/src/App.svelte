<script>
  import { onMount, onDestroy } from "svelte";

  /** @type {HTMLCanvasElement} */
  let mainCanvas;
  /** @type {CanvasRenderingContext2D} */
  let mainCtx;

  /** @type {HTMLCanvasElement} */
  let fovCanvas;
  /** @type {CanvasRenderingContext2D} */
  let fovCtx;
  let ws;

  const CELL_SIZE = 4;
  const FOV_CELL_SIZE = 7;
  const FOV_SIZE = 25;

  let isConnected = false;
  let latestState = null;

  let agentAngle = 0;
  let agentPrevPos = { x: null, y: null };

  onMount(() => {
    connectWebSocket();
  });

  function connectWebSocket() {
    ws = new WebSocket("ws://127.0.0.1:3000/ws");
    ws.onopen = () => (isConnected = true);
    ws.onclose = () => {
      isConnected = false;
      setTimeout(connectWebSocket, 1000);
    };
    ws.onmessage = (event) => {
      const state = JSON.parse(event.data);
      latestState = state;
      renderGlobalView(state);
      renderFOVView(state.fov);
    };
  }

  function triggerReset() {
    if (ws && isConnected) {
      ws.send(JSON.stringify({ command: "reset" }));
    }
  }

  function drawPacman(ctx, x, y, radius, color, timeOffset, angle = 0) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    const mouthAngle =
      Math.abs(Math.sin(Date.now() / 150 + timeOffset)) * (Math.PI / 4);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(0, 0, radius, mouthAngle, 2 * Math.PI - mouthAngle);
    ctx.lineTo(0, 0);
    ctx.fill();

    // Eyes
    ctx.fillStyle = "#11111b";
    const eyeSize = radius / 4;
    ctx.fillRect(-radius / 3, -radius / 1.5, eyeSize, eyeSize);

    ctx.restore();
  }

  function renderGlobalView(state) {
    if (!state || !state.width || !mainCanvas) return;
    if (!mainCtx) mainCtx = mainCanvas.getContext("2d"); // Fetch context here

    mainCanvas.width = state.width * CELL_SIZE;
    mainCanvas.height = state.height * CELL_SIZE;

    mainCtx.fillStyle = "#1e1e2e";
    mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);

    // Solid Static Walls
    mainCtx.fillStyle = "#45475a";
    for (let y = 0; y < state.height; y++) {
      for (let x = 0; x < state.width; x++) {
        if (state.static_grid[y * state.width + x] === 1) {
          // Add 0.5 to prevent anti-aliasing gaps that look like borders
          mainCtx.fillRect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            CELL_SIZE + 0.5,
            CELL_SIZE + 0.5,
          );
        }
      }
    }

    // Blinking Red Goal
    const pulse = (Math.sin(Date.now() / 150) + 1) / 2;
    const gx = state.goal_pos[0] * CELL_SIZE + CELL_SIZE / 2;
    const gy = state.goal_pos[1] * CELL_SIZE + CELL_SIZE / 2;

    const label = "GOAL";
    mainCtx.font = "20px 'VT323', monospace";
    const textMetrics = mainCtx.measureText(label);
    const padding = 8;
    const boxWidth = textMetrics.width + padding * 2;
    const boxHeight = 20 + padding; // approx height for 20px VT323 font
    const bx = gx - boxWidth / 2;
    const by = gy - boxHeight / 2;

    // Grey Box Background
    mainCtx.fillStyle = `rgba(69, 71, 90, ${0.4 + pulse * 0.4})`; // Catppuccin Surface0 (#45475a)
    mainCtx.fillRect(bx, by, boxWidth, boxHeight);

    // Blinking Text
    mainCtx.fillStyle = `rgba(243, 139, 168, ${0.4 + pulse * 0.6})`; // Catppuccin Red (#f38ba8)
    mainCtx.fillText(label, bx + padding, by + boxHeight - padding - 1);

    // Active Obstacles (Snakes & Boulders)
    state.obstacles.forEach((obs) => {
      if (obs.Boulder) {
        const bw = obs.Boulder.width * CELL_SIZE;
        const bh = obs.Boulder.height * CELL_SIZE;
        const bdAngle = Math.atan2(obs.Boulder.vy, obs.Boulder.vx);
        drawPacman(
          mainCtx,
          obs.Boulder.x * CELL_SIZE + bw / 2,
          obs.Boulder.y * CELL_SIZE + bh / 2,
          CELL_SIZE * 4, // Making radius CELL_SIZE * 4
          "#f38ba8", // Red color for boulders
          Date.now() / 150 + obs.Boulder.x, // Offset mouth animation
          bdAngle,
        );
      } else if (obs.Snake) {
        mainCtx.fillStyle = "#cba6f7";
        obs.Snake.body.forEach((segment) => {
          mainCtx.fillRect(
            segment[0] * CELL_SIZE,
            segment[1] * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
          );
        });
      }
    });

    const ax = state.agent_pos[0];
    const ay = state.agent_pos[1];
    const centerX = ax * CELL_SIZE + CELL_SIZE / 2;
    const centerY = ay * CELL_SIZE + CELL_SIZE / 2;

    // Fading LiDAR Rays
    const directions = [
      [0, 1],
      [0, -1],
      [1, 0],
      [-1, 0],
      [1, 1],
      [1, -1],
      [-1, 1],
      [-1, -1],
    ];

    directions.forEach((dir, i) => {
      // Extend distance by multiplying by a factor (e.g. 1.5)
      const dist = state.lidar[i] * 1.5;
      const endX = (ax + dir[0] * dist) * CELL_SIZE + CELL_SIZE / 2;
      const endY = (ay + dir[1] * dist) * CELL_SIZE + CELL_SIZE / 2;

      const grad = mainCtx.createLinearGradient(centerX, centerY, endX, endY);
      grad.addColorStop(0, "rgba(245, 194, 231, 0.4)"); // Pink, dimmer
      grad.addColorStop(1, "rgba(245, 194, 231, 0.0)");

      mainCtx.strokeStyle = grad;
      mainCtx.lineWidth = 0.5; // Thinner
      mainCtx.beginPath();
      mainCtx.moveTo(centerX, centerY);
      mainCtx.lineTo(endX, endY);
      mainCtx.stroke();
    });

    if (
      agentPrevPos.x !== null &&
      (agentPrevPos.x !== ax || agentPrevPos.y !== ay)
    ) {
      const dx = ax - agentPrevPos.x;
      const dy = ay - agentPrevPos.y;
      agentAngle = Math.atan2(dy, dx);
    }
    agentPrevPos.x = ax;
    agentPrevPos.y = ay;

    // Cute Agent (Blue Pacman)
    drawPacman(
      mainCtx,
      centerX,
      centerY,
      CELL_SIZE * 4, // Making radius CELL_SIZE * 4
      "#89dceb", // Blue agent
      0, // Fixed timeOffset base
      agentAngle, // Agent direction
    );

    // Draw Game Over Screen over canvas
    if (state.game_state === "GameOver") {
      mainCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
      mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);

      mainCtx.fillStyle = "#f38ba8"; // Red
      mainCtx.font = "80px 'VT323', monospace";
      mainCtx.textAlign = "center";
      mainCtx.fillText(
        "GAME OVER",
        mainCanvas.width / 2,
        mainCanvas.height / 2 - 20,
      );

      mainCtx.fillStyle = "#a6e3a1"; // Green
      mainCtx.font = "30px 'VT323', monospace";
      mainCtx.fillText(
        "Navigation Terminated.",
        mainCanvas.width / 2,
        mainCanvas.height / 2 + 30,
      );
      mainCtx.textAlign = "left"; // Reset
    }
  }

  let selectedEntity = null;

  function handleCanvasClick(e) {
    if (!latestState || latestState.game_state !== "Setup") return;

    const rect = mainCanvas.getBoundingClientRect();

    // Correctly scale mouse coordinates ignoring black bars from object-fit: contain
    const canvasRatio = mainCanvas.width / mainCanvas.height;
    const rectRatio = rect.width / rect.height;

    let drawWidth = rect.width;
    let drawHeight = rect.height;
    let offsetX = 0;
    let offsetY = 0;

    if (canvasRatio > rectRatio) {
      drawHeight = rect.width / canvasRatio;
      offsetY = (rect.height - drawHeight) / 2;
    } else {
      drawWidth = rect.height * canvasRatio;
      offsetX = (rect.width - drawWidth) / 2;
    }

    const clickX = e.clientX - rect.left - offsetX;
    const clickY = e.clientY - rect.top - offsetY;

    // Out of inner canvas bounds
    if (clickX < 0 || clickX > drawWidth || clickY < 0 || clickY > drawHeight) {
      if (selectedEntity) selectedEntity = null; // Unselect on empty void click
      return;
    }

    const px = (clickX / drawWidth) * mainCanvas.width;
    const py = (clickY / drawHeight) * mainCanvas.height;

    const cx = Math.floor(px / CELL_SIZE);
    const cy = Math.floor(py / CELL_SIZE);

    // If we've already selected an entity, second click drops it
    if (selectedEntity) {
      if (ws && isConnected) {
        if (selectedEntity.type === "agent") {
          ws.send(JSON.stringify({ command: "move_agent", x: cx, y: cy }));
        } else if (selectedEntity.type === "goal") {
          ws.send(JSON.stringify({ command: "move_goal", x: cx, y: cy }));
        } else if (selectedEntity.type === "obstacle") {
          ws.send(
            JSON.stringify({
              command: "move_obstacle",
              id: selectedEntity.id,
              x: cx,
              y: cy,
            }),
          );
        }
      }
      selectedEntity = null; // Unselect after moving
      return;
    }

    // Helper to check pixel distance
    const checkHit = (gridX, gridY, radiusPx) => {
      const targetPx = gridX * CELL_SIZE + CELL_SIZE / 2;
      const targetPy = gridY * CELL_SIZE + CELL_SIZE / 2;
      return Math.hypot(px - targetPx, py - targetPy) <= radiusPx;
    };

    const HIT_RADIUS = Math.max(CELL_SIZE * 6, 24); // Solid 24-pixel minimum radius

    // Otherwise, try to select an entity
    if (
      checkHit(latestState.agent_pos[0], latestState.agent_pos[1], HIT_RADIUS)
    ) {
      selectedEntity = { label: "Agent", type: "agent" };
      return;
    }

    // Goal text box is a bit wider, so give it a larger radius
    if (
      checkHit(
        latestState.goal_pos[0],
        latestState.goal_pos[1],
        HIT_RADIUS * 1.5,
      )
    ) {
      selectedEntity = { label: "Goal", type: "goal" };
      return;
    }

    for (let i = 0; i < latestState.obstacles.length; i++) {
      let obs = latestState.obstacles[i];
      if (obs.Boulder) {
        const bx = obs.Boulder.x + obs.Boulder.width / 2;
        const by = obs.Boulder.y + obs.Boulder.height / 2;
        if (checkHit(bx, by, HIT_RADIUS)) {
          selectedEntity = { label: "Boulder", type: "obstacle", id: i };
          return;
        }
      } else if (obs.Snake && obs.Snake.body.length > 0) {
        const head = obs.Snake.body[0];
        if (checkHit(head[0], head[1], HIT_RADIUS)) {
          selectedEntity = { label: "Snake", type: "obstacle", id: i };
          return;
        }
      }
    }
  }

  function triggerStart() {
    if (ws && isConnected) {
      ws.send(JSON.stringify({ command: "start" }));
    }
  }

  function renderFOVView(fovArray) {
    if (!fovArray || !fovCanvas) return;
    if (!fovCtx) fovCtx = fovCanvas.getContext("2d");

    fovCanvas.width = FOV_SIZE * FOV_CELL_SIZE;
    fovCanvas.height = FOV_SIZE * FOV_CELL_SIZE;
    fovCtx.fillStyle = "#11111b";
    fovCtx.fillRect(0, 0, fovCanvas.width, fovCanvas.height);

    for (let y = 0; y < FOV_SIZE; y++) {
      for (let x = 0; x < FOV_SIZE; x++) {
        const val = fovArray[y * FOV_SIZE + x];
        if (val === 1) fovCtx.fillStyle = "#45475a";
        else if (val === 2) fovCtx.fillStyle = "#f38ba8";
        else continue;
        fovCtx.fillRect(
          x * FOV_CELL_SIZE,
          y * FOV_CELL_SIZE,
          FOV_CELL_SIZE,
          FOV_CELL_SIZE,
        );
      }
    }
    const center = Math.floor(FOV_SIZE / 2);
    fovCtx.fillStyle = "#89dceb";
    fovCtx.fillRect(
      center * FOV_CELL_SIZE,
      center * FOV_CELL_SIZE,
      FOV_CELL_SIZE,
      FOV_CELL_SIZE,
    );
  }
</script>

<main class="container">
  <div class="header">
    <div style="display: flex; align-items: center; gap: 15px;">
      <h1>Hack 60: Autonomous Navigation Engine</h1>
      <a
        href="/replay.html"
        class="nav-link"
        style="color: #89b4fa; text-decoration: none; font-weight: bold;"
        >View Replays</a
      >
    </div>
    <span class="status {isConnected ? 'online' : 'offline'}">
      {isConnected ? "Backend Connected (30Hz)" : "Reconnecting..."}
    </span>
  </div>

  <div class="dashboard">
    <div class="global-view" style="position: relative;">
      <h2>Global Simulation</h2>

      {#if latestState && latestState.game_state === "Setup"}
        <div class="setup-overlay">
          <h3>Map Setup Editor</h3>
          <p style="color: #f38ba8; font-weight: bold;">
            {selectedEntity
              ? `Moving ${selectedEntity.label}... Click anywhere to place.`
              : "Click an Agent, Goal, or Obstacle to move it."}
          </p>
          {#if selectedEntity}
            <button
              class="action-btn"
              style="background-color: #f38ba8; margin-bottom: 10px;"
              on:click={() => (selectedEntity = null)}>Cancel Move</button
            >
          {/if}
          <button class="action-btn start-btn" on:click={triggerStart}>
            START SEARCH
          </button>
        </div>
      {/if}

      {#if latestState && latestState.game_state === "GameOver"}
        <div class="gameover-overlay">
          <button class="action-btn" on:click={triggerReset}>
            Return to Setup
          </button>
        </div>
      {/if}

      <canvas
        bind:this={mainCanvas}
        on:click={handleCanvasClick}
        style={latestState && latestState.game_state === "Setup"
          ? "cursor: " + (selectedEntity ? "crosshair" : "pointer")
          : ""}
      ></canvas>
    </div>

    <div class="side-panel">
      <div class="fov-view">
        <h2>Local Field of View</h2>
        <canvas bind:this={fovCanvas}></canvas>
        <p>Neural Network Input Tensor</p>
      </div>

      <div class="controls">
        <h2>Statistics</h2>
        {#if latestState}
          <div class="stats-box">
            <p>
              <strong>Agent Pos:</strong> ({latestState.agent_pos[0]}, {latestState
                .agent_pos[1]})
            </p>
            <p>
              <strong>Goal Pos:</strong> ({latestState.goal_pos[0]}, {latestState
                .goal_pos[1]})
            </p>
            <hr class="divider" />
            <p>
              <strong>Grid Dimensions:</strong>
              {latestState.width}x{latestState.height} ({latestState.width *
                latestState.height} cells)
            </p>
            <p>
              <strong>Solid Walls:</strong>
              {latestState.static_grid.filter((c) => c === 1).length} cells
            </p>
            <p>
              <strong>Free Navigable:</strong>
              {latestState.static_grid.filter((c) => c === 0).length} cells
            </p>
            <hr class="divider" />
            <p>
              <strong>Active Obstacles:</strong>
              {latestState.obstacles.length}
            </p>
            <ul class="stats-list">
              <li>
                Snakes: {latestState.obstacles.filter((o) => !!o.Snake).length}
              </li>
              <li>
                Boulders (Red Pacmans): {latestState.obstacles.filter(
                  (o) => !!o.Boulder,
                ).length}
              </li>
            </ul>
            <hr class="divider" />
            <p><strong>LiDAR Distances (8 rays):</strong></p>
            <div class="lidar-grid">
              {#each latestState.lidar as ray, i}
                <div class="lidar-cell" title="Ray {i}">{ray.toFixed(1)}</div>
              {/each}
            </div>
          </div>
        {:else}
          <p>Waiting for data...</p>
        {/if}
      </div>

      <!-- Metrics Panel -->
      <div class="metrics">
        <h3>Metrics:</h3>
        {#if latestState}
          <p>Wins: {latestState.goal_reach_count}</p>
          <p>Collisions: {latestState.collision_count}</p>
          <p>Inference: {latestState.replanning_speed_ms.toFixed(2)} ms</p>
          <p>Obstacles: {latestState.obstacles.length} (density: 5)</p>
          <p>
            Free Cells: {latestState.static_grid.filter((c) => c === 0).length}
          </p>
          <p>
            Navigable %: {(
              (latestState.static_grid.filter((c) => c === 0).length /
                (latestState.width * latestState.height)) *
              100
            ).toFixed(1)}%
          </p>
        {:else}
          <p>Waiting...</p>
        {/if}
      </div>

      <div class="controls">
        <h2>Simulation Controls</h2>
        <button
          class="action-btn"
          on:click={triggerReset}
          disabled={!isConnected ||
            (latestState && latestState.game_state !== "Setup")}
        >
          Generate New Domain Layout
        </button>
      </div>
    </div>
  </div>
</main>

<style>
  @import url("https://fonts.googleapis.com/css2?family=VT323&display=swap");

  :global(body) {
    margin: 0;
    padding: 0;
    overflow: hidden;
    background-color: #11111b;
    color: #cdd6f4;
    font-family: "VT323", monospace;
    font-size: 1rem;
    height: 100vh;
    width: 100vw;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .container {
    padding: 15px;
    width: 100vw;
    height: 100vh;
    max-width: calc(100vh * 16 / 9);
    max-height: calc(100vw * 9 / 16);
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #313244;
    padding-bottom: 5px;
    margin-bottom: 10px;
    flex-shrink: 0;
  }
  .header h1 {
    margin: 0;
    font-size: 1.5rem;
  }
  .status {
    padding: 5px 10px;
    border-radius: 5px;
    font-weight: bold;
  }
  .online {
    background-color: rgba(166, 227, 161, 0.2);
    color: #a6e3a1;
  }
  .offline {
    background-color: rgba(243, 139, 168, 0.2);
    color: #f38ba8;
  }
  .dashboard {
    display: flex;
    gap: 15px;
    flex: 1;
    min-height: 0;
  }
  .global-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }
  .global-view h2 {
    flex-shrink: 0;
  }
  .global-view canvas {
    flex: 1;
    border: 2px solid #313244;
    border-radius: 8px;
    background-color: #1e1e2e;
    width: 100%;
    height: 100%;
    max-height: 100%;
    object-fit: contain;
    min-height: 0;
  }
  .fov-view {
    background-color: #1e1e2e;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #313244;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
  }
  .fov-view h2 {
    margin-bottom: 0;
    align-self: flex-start;
  }
  .fov-view canvas {
    border: 2px solid #313244;
    border-radius: 8px;
    background-color: #11111b;
    width: 120px;
    height: 120px;
    object-fit: contain;
    flex-shrink: 0;
  }
  .fov-view p {
    margin: 0;
    font-size: 0.9rem;
    color: #bac2de;
  }
  .side-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    flex: 0 0 300px;
    min-height: 0;
  }
  .controls {
    background-color: #1e1e2e;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #313244;
    display: flex;
    flex-direction: column;
    gap: 5px;
    flex-shrink: 0;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 5px 10px;
    background-color: #1e1e2e;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #313244;
    flex-shrink: 0;
  }
  .metrics h3 {
    grid-column: span 2;
    margin: 0 0 5px 0;
    color: #a6e3a1;
    font-size: 1.1rem;
  }
  .metrics p,
  .param-controls label {
    margin: 0;
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .stats-box {
    font-size: 1rem;
    line-height: 1.2;
  }
  .stats-box p {
    margin: 2px 0;
  }
  .divider {
    border: 0;
    height: 1px;
    background: #313244;
    margin: 5px 0;
  }
  .stats-list {
    margin: 2px 0 0 0;
    padding-left: 20px;
    color: #bac2de;
  }
  .lidar-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5px;
    text-align: center;
    margin-top: 5px;
  }
  .lidar-cell {
    background: #45475a;
    padding: 3px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.9rem;
    color: #f5c2e7;
  }
  .action-btn {
    font-family: "VT323", monospace;
    background-color: #89b4fa;
    color: #11111b;
    border: none;
    padding: 8px 15px;
    border-radius: 5px;
    font-weight: bold;
    cursor: pointer;
    width: 100%;
    font-size: 1.1rem;
    transition:
      transform 0.1s,
      background-color 0.2s;
  }
  .action-btn:hover:not(:disabled) {
    background-color: #b4befe;
    transform: scale(1.02);
  }
  .action-btn:active:not(:disabled) {
    transform: scale(0.98);
  }
  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .setup-overlay {
    position: absolute;
    top: 50px;
    right: 20px;
    background: rgba(30, 30, 46, 0.9);
    border: 2px dashed #b4befe;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    pointer-events: auto; /* Ensure buttons in here are clickable */
    z-index: 10;
  }
  .setup-overlay p {
    margin: 5px;
    font-size: 1.1rem;
    color: #f5c2e7;
  }
  .start-btn {
    background-color: #a6e3a1; /* Green for start */
    color: #11111b;
    margin-top: 10px;
  }
  .start-btn:hover:not(:disabled) {
    background-color: #89dceb;
  }
  .gameover-overlay {
    position: absolute;
    top: 55%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 10;
    pointer-events: auto;
  }
  h2 {
    font-size: 1.2rem;
    margin-top: 0;
    color: #b4befe;
  }
</style>
