import asyncio
from playwright.async_api import async_playwright
import os
import glob
import re
import subprocess
import base64


async def record_video(page, replay_filename, text_title):
    print(f"Loading {replay_filename}...")
    await page.goto("http://localhost:5173/replay.html")
    await asyncio.sleep(2)

    await page.evaluate(f"window.loadReplayFile('{replay_filename}')")
    await asyncio.sleep(2)

    print(f"Recording {replay_filename}...")
    await page.evaluate("window.startRecording(30)")

    base64_data = None
    while not base64_data:
        base64_data = await page.evaluate("window['_recordedVideoBase64']")
        if not base64_data:
            await asyncio.sleep(0.5)

    await page.evaluate("window['_recordedVideoBase64'] = null")

    header, encoded = base64_data.split(",", 1)
    binary_data = base64.b64decode(encoded)

    # Security check for file size (180 MB = 180 * 1024 * 1024 bytes)
    max_size = 188743680
    if len(binary_data) > max_size:
        print(
            f"ERROR: {replay_filename} data ({len(binary_data)} bytes) exceeds 180MB limit. Skipping to prevent crash."
        )
        return

    webm_path = f"temp_{replay_filename}.webm"

    with open(webm_path, "wb") as f:
        f.write(binary_data)
        f.flush()
        os.fsync(f.fileno())

    file_size = os.path.getsize(webm_path)
    if file_size == 0:
        print(f"ERROR: {webm_path} is empty. Skipping.")
        os.remove(webm_path)
        return

    print(f"WebM written: {file_size} bytes")

    final_mp4 = f"video_{replay_filename}.mp4"
    print(f"Converting to {final_mp4} with text overlay...")

    text_filter = f"drawtext=text='{text_title}':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:alpha='if(lt(t,1),t/1,if(lt(t,2.5),1,if(lt(t,3.5),3.5-t,0)))'"
    intro_mp4 = f"intro_{replay_filename}.mp4"

    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        webm_path,
    ]
    try:
        probe_result = (
            subprocess.check_output(probe_cmd, stderr=subprocess.PIPE)
            .decode("utf-8")
            .strip()
        )
        width_height = probe_result.replace(",", "x")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffprobe failed for {webm_path}: {e.stderr.decode().strip()}")
        os.remove(webm_path)
        return

    if not width_height:
        print(f"ERROR: Could not determine resolution of {webm_path}. Skipping.")
        os.remove(webm_path)
        return

    print(f"Detected resolution: {width_height}")

    main_mp4 = f"main_{replay_filename}.mp4"
    list_path = f"list_{replay_filename}.txt"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                webm_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-y",
                main_mp4,
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={width_height}:d=3.5",
                "-vf",
                text_filter,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-y",
                intro_mp4,
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )

        with open(list_path, "w") as f:
            f.write(f"file '{intro_mp4}'\n")
            f.write(f"file '{main_mp4}'\n")

        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                "-y",
                final_mp4,
            ],
            check=True,
            stderr=subprocess.DEVNULL,
        )
        print(f"Finished {final_mp4}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffmpeg failed for {replay_filename}: {e}")
    finally:
        for f_path in [intro_mp4, main_mp4, webm_path, list_path]:
            if os.path.exists(f_path):
                os.remove(f_path)


async def main():
    replay_files = glob.glob("frontend/public/replays/replay_step_*.json")

    if not replay_files:
        print("No replay files found.")
        return

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        print("Navigating to http://localhost:5173/replay.html")
        try:
            await page.goto("http://localhost:5173/replay.html")
            await asyncio.sleep(2)
        except Exception as e:
            print(
                "Failed to reach localhost:5173. Please make sure the 'npm run dev' server is running."
            )
            return

        for path in sorted(replay_files):
            filename = os.path.basename(path)
            match = re.search(r"replay_step_(\d+)\.json", filename)
            if match:
                episodes = match.group(1)
                title = f"after {episodes} training episodes"
                await record_video(page, filename, title)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
