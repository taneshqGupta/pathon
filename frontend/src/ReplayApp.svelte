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

  const CELL_SIZE = 4;
  const FOV_CELL_SIZE = 7;
  const FOV_SIZE = 25;

  let frames = [];
  let currentFrameIndex = 0;
  let totalFrames = 0;
  let currentChunkStart = 0;
  const CHUNK_SIZE = 1000;
  let isPlaying = false;
  let playSpeed = 30; // ms per frame
  let intervalId;
  let selectedReplay = "replay_step_100000.json";
  let isLoading = false;
  let isFetchingChunk = false;

  const availableReplays = [
    "replay_step_50000.json",
    "replay_step_100000.json",
    "replay_step_150000.json",
    "replay_step_200000.json",
    "replay_step_250000.json",
    "replay_step_300000.json",
    "replay_step_350000.json",
    "replay_step_400020.json",
    "replay_step_900000.json",
    "replay_step_700020.json",
    "replay_step_850020.json",
    "replay_step_800010.json",
    "replay_step_750000.json",
    "replay_step_950010.json",
    "replay_step_1300020.json",
    "replay_step_1250010.json",
    "replay_step_1200000.json",
    "replay_step_1500000.json",
    "replay_step_1550010.json",
    "replay_step_1600020.json",
    "replay_step_1650000.json",
    "replay_step_1700010.json",
    "replay_step_1750020.json",
    "replay_step_1800000.json",
    "replay_step_1850010.json",
    "replay_step_1900020.json",
    "replay_step_1950000.json",
    "replay_step_1400010.json",
    

  ];

  $: localIndex = currentFrameIndex - currentChunkStart;
  $: currentState = (frames.length > 0 && localIndex >= 0 && localIndex < frames.length) ? frames[localIndex] : null;

  let agentAngle = 0;
  let agentPrevPos = { x: null, y: null };

  onMount(() => {
    loadReplay();
  });

  onDestroy(() => {
    stopPlayback();
  });

  async function fetchChunk(start) {
    isFetchingChunk = true;
    try {
      const res = await fetch(`http://localhost:8000/api/replay?file=${selectedReplay}&start=${start}&size=${CHUNK_SIZE}`);
      if (res.ok) {
        const data = await res.json();
        frames = data.frames;
        totalFrames = data.totalFrames;
        currentChunkStart = start;
        if (frames.length > 0) {
          renderCurrentFrame();
        }
      } else {
        console.error("Failed to fetch chunk");
      }
    } catch (e) {
      console.error(e);
    } finally {
      isFetchingChunk = false;
    }
  }

  async function loadReplay() {
    isLoading = true;
    stopPlayback();
    currentFrameIndex = 0;
    frames = [];
    currentChunkStart = 0;
    await fetchChunk(0);
    isLoading = false;
  }

  function togglePlayback() {
    isPlaying = !isPlaying;
    if (isPlaying) {
      startPlayback();
    } else {
      stopPlayback();
    }
  }

  function startPlayback() {
    if (intervalId) clearInterval(intervalId);
    intervalId = setInterval(async () => {
      if (isFetchingChunk) return; // Pause playback while fetching

      if (currentFrameIndex < totalFrames - 1) {
        currentFrameIndex++;
        
        // If we stepped out of our loaded chunk, fetch the next one
        if (currentFrameIndex >= currentChunkStart + frames.length) {
          await fetchChunk(currentFrameIndex);
        } else {
          renderCurrentFrame();
        }
      } else {
        stopPlayback();
        isPlaying = false;
      }
    }, playSpeed);
  }

  function stopPlayback() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  $: if (isPlaying && intervalId && !isFetchingChunk) {
    // Restart interval if playspeed changes
    startPlayback();
  }

  async function handleSeek(e) {
    currentFrameIndex = parseInt(e.target.value, 10);
    // If the seeked index is outside our current chunk, fetch it
    if (currentFrameIndex < currentChunkStart || currentFrameIndex >= currentChunkStart + frames.length) {
      // Fetch a new chunk centered around the seek position (or just starting at the seek position)
      await fetchChunk(currentFrameIndex);
    } else {
      renderCurrentFrame();
    }
  }

  function renderCurrentFrame() {
    if (!currentState) return;
    renderGlobalView(currentState);
    renderFOVView(currentState.fov);
  }

  function drawPacman(ctx, x, y, radius, color, timeOffset, angle = 0) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    const mouthAngle = Math.abs(Math.sin(Date.now() / 150 + timeOffset)) * (Math.PI / 4);
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
    if (!mainCtx) mainCtx = mainCanvas.getContext("2d");

    mainCanvas.width = state.width * CELL_SIZE;
    mainCanvas.height = state.height * CELL_SIZE;

    mainCtx.fillStyle = "#1e1e2e";
    mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);

    mainCtx.fillStyle = "#45475a";
    for (let y = 0; y < state.height; y++) {
      for (let x = 0; x < state.width; x++) {
        if (state.static_grid[y * state.width + x] === 1) {
          mainCtx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE + 0.5, CELL_SIZE + 0.5);
        }
      }
    }

    const pulse = (Math.sin(Date.now() / 150) + 1) / 2;
    const gx = state.goal_pos[0] * CELL_SIZE + CELL_SIZE / 2;
    const gy = state.goal_pos[1] * CELL_SIZE + CELL_SIZE / 2;

    const label = "GOAL";
    mainCtx.font = "20px 'VT323', monospace";
    const textMetrics = mainCtx.measureText(label);
    const padding = 8;
    const boxWidth = textMetrics.width + padding * 2;
    const boxHeight = 20 + padding;
    const bx = gx - boxWidth / 2;
    const by = gy - boxHeight / 2;

    mainCtx.fillStyle = `rgba(69, 71, 90, ${0.4 + pulse * 0.4})`;
    mainCtx.fillRect(bx, by, boxWidth, boxHeight);
    mainCtx.fillStyle = `rgba(243, 139, 168, ${0.4 + pulse * 0.6})`;
    mainCtx.fillText(label, bx + padding, by + boxHeight - padding - 1);

    if (state.obstacles) {
      state.obstacles.forEach((obs) => {
        if (obs.Boulder) {
          const bw = obs.Boulder.width * CELL_SIZE;
          const bh = obs.Boulder.height * CELL_SIZE;
          const bdAngle = Math.atan2(obs.Boulder.vy, obs.Boulder.vx);
          drawPacman(mainCtx, obs.Boulder.x * CELL_SIZE + bw / 2, obs.Boulder.y * CELL_SIZE + bh / 2, CELL_SIZE * 4, "#f38ba8", Date.now() / 150 + obs.Boulder.x, bdAngle);
        } else if (obs.Snake) {
          mainCtx.fillStyle = "#cba6f7";
          obs.Snake.body.forEach((segment) => {
            mainCtx.fillRect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE);
          });
        }
      });
    }

    const ax = state.agent_pos[0];
    const ay = state.agent_pos[1];
    const centerX = ax * CELL_SIZE + CELL_SIZE / 2;
    const centerY = ay * CELL_SIZE + CELL_SIZE / 2;

    const directions = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];
    directions.forEach((dir, i) => {
      const dist = state.lidar[i] * 1.5;
      const endX = (ax + dir[0] * dist) * CELL_SIZE + CELL_SIZE / 2;
      const endY = (ay + dir[1] * dist) * CELL_SIZE + CELL_SIZE / 2;
      const grad = mainCtx.createLinearGradient(centerX, centerY, endX, endY);
      grad.addColorStop(0, "rgba(245, 194, 231, 0.4)");
      grad.addColorStop(1, "rgba(245, 194, 231, 0.0)");
      mainCtx.strokeStyle = grad;
      mainCtx.lineWidth = 0.5;
      mainCtx.beginPath();
      mainCtx.moveTo(centerX, centerY);
      mainCtx.lineTo(endX, endY);
      mainCtx.stroke();
    });

    if (agentPrevPos.x !== null && (agentPrevPos.x !== ax || agentPrevPos.y !== ay)) {
      const dx = ax - agentPrevPos.x;
      const dy = ay - agentPrevPos.y;
      agentAngle = Math.atan2(dy, dx);
    }
    agentPrevPos.x = ax;
    agentPrevPos.y = ay;

    drawPacman(mainCtx, centerX, centerY, CELL_SIZE * 4, "#89dceb", 0, agentAngle);

    if (state.game_state === "GameOver" || state.game_state === "Setup") {
      mainCtx.fillStyle = "rgba(0, 0, 0, 0.7)";
      mainCtx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
      mainCtx.fillStyle = "#f38ba8";
      mainCtx.font = "80px 'VT323', monospace";
      mainCtx.textAlign = "center";
      mainCtx.fillText(state.game_state, mainCanvas.width / 2, mainCanvas.height / 2 - 20);
      mainCtx.textAlign = "left";
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
        fovCtx.fillRect(x * FOV_CELL_SIZE, y * FOV_CELL_SIZE, FOV_CELL_SIZE, FOV_CELL_SIZE);
      }
    }
    const center = Math.floor(FOV_SIZE / 2);
    fovCtx.fillStyle = "#89dceb";
    fovCtx.fillRect(center * FOV_CELL_SIZE, center * FOV_CELL_SIZE, FOV_CELL_SIZE, FOV_CELL_SIZE);
  }
</script>

<main class="container">
  <div class="header">
    <h1>Replay Viewer</h1>
    <div class="replay-selector">
      <select bind:value={selectedReplay} disabled={isLoading}>
        {#each availableReplays as rep}
          <option value={rep}>{rep}</option>
        {/each}
      </select>
      <button on:click={loadReplay} disabled={isLoading} class="action-btn" style="width: auto;">
        {isLoading ? 'Loading...' : 'Load'}
      </button>
      <a href="/" class="nav-link">Go to Sim</a>
    </div>
  </div>

  <div class="dashboard">
    <div class="global-view">
      <canvas bind:this={mainCanvas}></canvas>
    </div>

    <div class="side-panel">
      <div class="fov-view">
        <h2>Local Field of View</h2>
        <canvas bind:this={fovCanvas}></canvas>
      </div>

      <div class="controls">
        <h2>Playback Controls</h2>
        <button class="action-btn" on:click={togglePlayback} disabled={totalFrames === 0}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <div class="param-controls" style="flex-direction: column; gap: 10px; margin-top: 10px;">
          <label>
            Speed (ms/frame):
            <input type="range" min="10" max="500" step="10" bind:value={playSpeed} disabled={isFetchingChunk} />
            {playSpeed}ms
          </label>
          <label>
            Frame: {currentFrameIndex} / {totalFrames > 0 ? totalFrames - 1 : 0} {isFetchingChunk ? '(Loading...)' : ''}
            <input
              type="range"
              min="0"
              max={totalFrames > 0 ? totalFrames - 1 : 0}
              value={currentFrameIndex}
              on:input={handleSeek}
              style="width: 100%"
              disabled={isFetchingChunk}
            />
          </label>
        </div>
      </div>

      <div class="controls">
        <h2>Frame Details</h2>
        {#if currentState}
          <div class="stats-box">
            <p><strong>Agent Pos:</strong> ({currentState.agent_pos[0]}, {currentState.agent_pos[1]})</p>
            <p><strong>Goal Pos:</strong> ({currentState.goal_pos[0]}, {currentState.goal_pos[1]})</p>
          </div>
        {:else}
          <p>No frame loaded.</p>
        {/if}
      </div>
    </div>
  </div>
</main>

<style>
  @import url("https://fonts.googleapis.com/css2?family=VT323&display=swap");

  :global(body) {
    margin: 0; padding: 0; overflow: hidden;
    background-color: #11111b; color: #cdd6f4;
    font-family: "VT323", monospace; font-size: 1rem;
    height: 100vh; width: 100vw;
    display: flex; justify-content: center; align-items: center;
  }
  .container {
    padding: 10px; width: 100%; max-width: 100vw; max-height: 100vh;
    box-sizing: border-box; display: flex; flex-direction: column; justify-content: center;
    aspect-ratio: 16/9;
  }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #313244; padding-bottom: 5px; margin-bottom: 10px;
  }
  .header h1 { margin: 0; font-size: 1.5rem; }
  .replay-selector { display: flex; gap: 10px; align-items: center; }
  .replay-selector select { 
    background-color: #1e1e2e; color: #cdd6f4; 
    border: 1px solid #313244; border-radius: 5px; 
    padding: 5px; font-family: "VT323", monospace; font-size: 1.1rem;
  }
  .nav-link { 
      color: #89b4fa; text-decoration: none; 
      font-weight: bold; margin-left: 10px; 
  }
  .nav-link:hover { text-decoration: underline; color: #b4befe; }
  .dashboard { display: flex; gap: 15px; flex: 1; min-height: 0; }
  .global-view { flex: 1 1 auto; display: flex; flex-direction: column; height: 100%; min-width: 0; }
  .global-view canvas, .fov-view canvas {
    border: 2px solid #313244; border-radius: 8px; background-color: #1e1e2e;
    max-height: 100%; max-width: 100%; object-fit: contain;
  }
  .side-panel { display: flex; flex-direction: column; gap: 10px; flex: 0 0 320px; max-height: 100%; overflow-y: hidden; }
  .controls { background-color: #1e1e2e; padding: 10px; border-radius: 8px; border: 1px solid #313244; display: flex; flex-direction: column; gap: 5px; }
  h2 { font-size: 1.2rem; margin-top: 0; color: #b4befe; }
  .action-btn { font-family: "VT323", monospace; background-color: #89b4fa; color: #11111b; border: none; padding: 8px 15px; border-radius: 5px; font-weight: bold; cursor: pointer; width: 100%; font-size: 1.1rem; transition: transform 0.1s, background-color 0.2s; }
  .action-btn:hover:not(:disabled) { background-color: #b4befe; transform: scale(1.02); }
  .action-btn:active:not(:disabled) { transform: scale(0.98); }
  .action-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>