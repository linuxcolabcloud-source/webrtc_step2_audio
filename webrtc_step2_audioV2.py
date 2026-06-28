#!/usr/bin/env python3
"""
WebRTC Remote Desktop - Bước 2: Audio (PulseAudio)
===================================================
Tự cài và khởi động:
- Xvfb (màn hình ảo)
- Xfce4 (desktop)
- PulseAudio (âm thanh)
- WebRTC server (stream)
- Cloudflare Tunnel (link public)
"""

import asyncio
import fractions
import json
import logging
import os
import subprocess
import threading
import time
import sys
from typing import Set

import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from av import VideoFrame, AudioFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-desktop")

pcs: Set[RTCPeerConnection] = set()

DISPLAY    = os.environ.get("DISPLAY", ":99")
SCREEN_W   = int(os.environ.get("SCREEN_W", "1920"))
SCREEN_H   = int(os.environ.get("SCREEN_H", "1080"))
TARGET_FPS = int(os.environ.get("TARGET_FPS", "60"))
BITRATE    = os.environ.get("BITRATE", "8M")
SAMPLE_RATE = 48000  # Hz — chuẩn WebRTC
CHANNELS    = 2      # stereo

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO TRACK (giữ nguyên từ Bước 1)
# ═══════════════════════════════════════════════════════════════════════════════
class NvencCaptureTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 90000)
        self._frame_q   = asyncio.Queue(maxsize=4)
        self._running   = True
        self._thread    = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        frame_size = 1280 * 720 * 3 // 2
        while self._running:
            try:
                cmd = [
                    "ffmpeg", "-loglevel", "error",
                    "-f", "x11grab",
                    "-video_size", f"{SCREEN_W}x{SCREEN_H}",
                    "-framerate", str(TARGET_FPS),
                    "-i", DISPLAY,
                    "-vf", "scale=1280:720",
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv420p",
                    "pipe:1"
                ]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=frame_size * 2
                )
                while self._running:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((720 * 3 // 2, 1280))
                    try:
                        self._frame_q.put_nowait(yuv)
                    except asyncio.QueueFull:
                        try:
                            self._frame_q.get_nowait()
                            self._frame_q.put_nowait(yuv)
                        except:
                            pass
                proc.kill()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Video capture error: {e}")
                time.sleep(1)

    async def recv(self):
        pts = int(self._timestamp * 90000 / TARGET_FPS)
        self._timestamp += 1
        try:
            yuv = await asyncio.wait_for(self._frame_q.get(), timeout=2.0 / TARGET_FPS)
        except asyncio.TimeoutError:
            yuv = np.zeros((720 * 3 // 2, 1280), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(yuv, format="yuv420p")
        frame.pts = pts
        frame.time_base = self._time_base
        return frame

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO TRACK — PulseAudio → ffmpeg → WebRTC
# ═══════════════════════════════════════════════════════════════════════════════
class PulseAudioTrack(AudioStreamTrack):
    """
    Capture audio từ PulseAudio monitor (output của hệ thống)
    ffmpeg pulse → raw PCM s16le → AudioFrame → WebRTC
    
    Dùng 'default' monitor source = nghe tất cả âm thanh đang phát
    """
    kind = "audio"

    # Số samples mỗi frame (10ms @ 48kHz = 480 samples)
    SAMPLES_PER_FRAME = 480

    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, SAMPLE_RATE)
        self._audio_q   = asyncio.Queue(maxsize=10)
        self._running   = True
        self._volume    = 1.0  # 0.0 → 1.0

        # Tìm PulseAudio monitor source
        self._pulse_source = self._get_pulse_monitor()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Audio: PulseAudio source = {self._pulse_source}")

    def _get_pulse_monitor(self):
        """
        Lấy tên monitor source của PulseAudio
        Monitor = capture lại những gì đang được phát ra loa
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                # Tìm source có đuôi .monitor
                if ".monitor" in line:
                    return line.split()[1]
        except:
            pass
        return "default"

    def _capture_loop(self):
        """
        ffmpeg đọc từ PulseAudio → raw PCM s16le stereo 48kHz
        Mỗi chunk = SAMPLES_PER_FRAME * 2 channels * 2 bytes = 1920 bytes
        """
        chunk_size = self.SAMPLES_PER_FRAME * CHANNELS * 2  # s16le = 2 bytes

        while self._running:
            try:
                cmd = [
                    "ffmpeg", "-loglevel", "error",
                    # Input: PulseAudio
                    "-f", "pulse",
                    "-i", self._pulse_source,
                    # Output: raw PCM
                    "-f", "s16le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", str(CHANNELS),
                    "pipe:1"
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=chunk_size * 4
                )
                logger.info("ffmpeg audio capture started")

                while self._running:
                    raw = proc.stdout.read(chunk_size)
                    if len(raw) < chunk_size:
                        break

                    # Convert bytes → numpy int16
                    pcm = np.frombuffer(raw, dtype=np.int16)

                    # Áp volume
                    if self._volume != 1.0:
                        pcm = (pcm * self._volume).astype(np.int16)

                    # Reshape: (samples, channels)
                    pcm = pcm.reshape(-1, CHANNELS)

                    try:
                        self._audio_q.put_nowait(pcm)
                    except asyncio.QueueFull:
                        try:
                            self._audio_q.get_nowait()
                            self._audio_q.put_nowait(pcm)
                        except:
                            pass

                proc.kill()
                logger.warning("ffmpeg audio died, restarting...")
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Audio capture error: {e}")
                time.sleep(1)

    async def recv(self):
        pts = self._timestamp
        self._timestamp += self.SAMPLES_PER_FRAME

        try:
            pcm = await asyncio.wait_for(
                self._audio_q.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            # Silence nếu không có audio
            pcm = np.zeros((self.SAMPLES_PER_FRAME, CHANNELS), dtype=np.int16)

        # Tạo AudioFrame
        # layout: stereo, format: s16, sample_rate: 48000
        frame = AudioFrame.from_ndarray(
            pcm.T,  # aiortc cần (channels, samples)
            format="s16",
            layout="stereo"
        )
        frame.pts = pts
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = self._time_base
        return frame

    def set_volume(self, vol: float):
        """0.0 = mute, 1.0 = 100%"""
        self._volume = max(0.0, min(1.0, vol))

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# HTML CLIENT — thêm audio controls + volume slider
# ═══════════════════════════════════════════════════════════════════════════════
HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WebRTC Remote Desktop</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0f0f0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  #toolbar {
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }
  #toolbar h1 { font-size: 14px; font-weight: 600; color: #fff; }

  #status {
    font-size: 12px; padding: 3px 10px;
    border-radius: 20px; background: #2a2a2a; color: #888;
    transition: all 0.3s;
  }
  #status.connecting { background: #2a1f00; color: #f0a500; }
  #status.connected  { background: #0a2a0a; color: #4caf50; }
  #status.error      { background: #2a0a0a; color: #f44336; }

  /* Volume control */
  #volume-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: 8px;
  }
  #volume-wrap span { font-size: 16px; cursor: pointer; }
  #volume-slider {
    -webkit-appearance: none;
    width: 80px; height: 4px;
    background: #444; border-radius: 2px; outline: none;
    cursor: pointer;
  }
  #volume-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    background: #2563eb; border-radius: 50%;
  }
  #volume-label { font-size: 11px; color: #666; width: 28px; }

  .btn {
    background: #2a2a2a; color: #ccc;
    border: none; padding: 6px 12px;
    border-radius: 6px; font-size: 13px; cursor: pointer;
    transition: background 0.2s;
  }
  .btn:hover { background: #383838; }
  .btn.primary { background: #2563eb; color: #fff; }
  .btn.primary:hover { background: #1d4ed8; }
  .btn:disabled { background: #222; color: #555; cursor: not-allowed; }

  #video-container {
    flex: 1; display: flex;
    align-items: center; justify-content: center;
    overflow: hidden; position: relative;
  }
  #remote-video { max-width: 100%; max-height: 100%; display: block; cursor: none; }

  #overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 16px; background: #0f0f0f;
  }
  #overlay svg { opacity: 0.25; }
  #overlay p { color: #555; font-size: 14px; }
  #overlay.hidden { display: none; }

  #stats {
    position: absolute; top: 8px; left: 8px;
    background: rgba(0,0,0,0.65); color: #0f0;
    font-size: 11px; font-family: monospace;
    padding: 4px 8px; border-radius: 4px; display: none;
  }

  #toast {
    position: fixed; bottom: 20px; right: 20px;
    background: #1e1e1e; border: 1px solid #333;
    color: #ccc; padding: 8px 14px;
    border-radius: 8px; font-size: 12px;
    opacity: 0; transition: opacity 0.3s; pointer-events: none;
  }
  #toast.show { opacity: 1; }
</style>
</head>
<body>

<div id="toolbar">
  <h1>🖥 Remote Desktop</h1>
  <span id="status">Chưa kết nối</span>

  <div id="volume-wrap">
    <span id="mute-btn" title="Mute">🔊</span>
    <input id="volume-slider" type="range" min="0" max="100" value="80">
    <span id="volume-label">80%</span>
  </div>

  <button id="btn-fullscreen" class="btn">⛶</button>
  <button id="btn-connect" class="btn primary">Kết nối</button>
</div>

<div id="video-container">
  <video id="remote-video" autoplay playsinline></video>

  <div id="overlay">
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none"
         stroke="#fff" stroke-width="1.5">
      <rect x="2" y="3" width="20" height="14" rx="2"/>
      <path d="M8 21h8M12 17v4"/>
    </svg>
    <p>Nhấn <strong>Kết nối</strong> để bắt đầu</p>
  </div>

  <div id="stats"></div>
</div>

<div id="toast"></div>

<script>
let pc = null, dataChannel = null, muted = false;
const videoEl  = document.getElementById('remote-video');
const overlay  = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const btnConn  = document.getElementById('btn-connect');
const statsEl  = document.getElementById('stats');
const volSlider = document.getElementById('volume-slider');
const volLabel  = document.getElementById('volume-label');
const muteBtn   = document.getElementById('mute-btn');

function setStatus(t, c) { statusEl.textContent = t; statusEl.className = c||''; }
function showToast(m) {
  const t = document.getElementById('toast');
  t.textContent = m; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}
function sendInput(d) {
  if (dataChannel?.readyState === 'open') dataChannel.send(JSON.stringify(d));
}

// ── Volume ─────────────────────────────────────────────────────────────────
volSlider.oninput = () => {
  const v = volSlider.value / 100;
  videoEl.volume = v;
  volLabel.textContent = volSlider.value + '%';
  muteBtn.textContent = v === 0 ? '🔇' : v < 0.5 ? '🔉' : '🔊';
  sendInput({ type: 'volume', value: v });
};

muteBtn.onclick = () => {
  muted = !muted;
  videoEl.muted = muted;
  muteBtn.textContent = muted ? '🔇' : '🔊';
  sendInput({ type: 'volume', value: muted ? 0 : volSlider.value / 100 });
};

// ── Connect ────────────────────────────────────────────────────────────────
async function connect() {
  btnConn.disabled = true;
  setStatus('Đang kết nối...', 'connecting');

  pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });

  pc.ontrack = (e) => {
    if (e.track.kind === 'video') {
      videoEl.srcObject = e.streams[0];
      overlay.classList.add('hidden');
      setStatus('Đã kết nối', 'connected');
      btnConn.textContent = 'Ngắt';
      btnConn.disabled = false;
      btnConn.onclick = disconnect;
      startStats();
    }
  };

  dataChannel = pc.createDataChannel('input', { ordered: true });

  pc.onconnectionstatechange = () => {
    if (['disconnected','failed','closed'].includes(pc.connectionState)) {
      setStatus('Mất kết nối', 'error');
      overlay.classList.remove('hidden');
      btnConn.textContent = 'Kết nối lại';
      btnConn.disabled = false;
      btnConn.onclick = connect;
    }
  };

  try {
    const offer = await pc.createOffer({ offerToReceiveVideo: true, offerToReceiveAudio: true });
    await pc.setLocalDescription(offer);
    const res = await fetch('/offer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type })
    });
    const answer = await res.json();
    await pc.setRemoteDescription(new RTCSessionDescription(answer));
  } catch(err) {
    setStatus('Lỗi: ' + err.message, 'error');
    btnConn.disabled = false;
  }
}

function disconnect() {
  pc?.close(); pc = null;
  videoEl.srcObject = null;
  overlay.classList.remove('hidden');
  setStatus('Đã ngắt', '');
  btnConn.textContent = 'Kết nối';
  btnConn.onclick = connect;
  statsEl.style.display = 'none';
}

// ── Mouse ──────────────────────────────────────────────────────────────────
function relPos(e) {
  const r = videoEl.getBoundingClientRect();
  return { x: (e.clientX-r.left)/r.width, y: (e.clientY-r.top)/r.height };
}
videoEl.addEventListener('mousemove',  e => sendInput({ type:'mousemove',  ...relPos(e) }));
videoEl.addEventListener('mousedown',  e => sendInput({ type:'mousedown',  button:e.button, ...relPos(e) }));
videoEl.addEventListener('mouseup',    e => sendInput({ type:'mouseup',    button:e.button, ...relPos(e) }));
videoEl.addEventListener('wheel',      e => { e.preventDefault(); sendInput({ type:'wheel', dx:e.deltaX, dy:e.deltaY }); }, { passive:false });
videoEl.addEventListener('contextmenu', e => e.preventDefault());

// ── Keyboard ───────────────────────────────────────────────────────────────
const KEY_MAP = {
  ' ':'space','Enter':'Return','Backspace':'BackSpace','Delete':'Delete',
  'Escape':'Escape','Tab':'Tab','ArrowUp':'Up','ArrowDown':'Down',
  'ArrowLeft':'Left','ArrowRight':'Right'
};
document.addEventListener('keydown', e => {
  if (!dataChannel) return;
  e.preventDefault();
  sendInput({ type:'keydown', key: KEY_MAP[e.key]||e.key, code:e.code });
});
document.addEventListener('keyup', e => {
  if (!dataChannel) return;
  sendInput({ type:'keyup', key: KEY_MAP[e.key]||e.key, code:e.code });
});

// ── Clipboard ──────────────────────────────────────────────────────────────
document.addEventListener('paste', e => {
  const text = e.clipboardData.getData('text');
  if (text) { sendInput({ type:'clipboard', text }); showToast('📋 Paste: ' + text.slice(0,30)); }
});
document.addEventListener('keydown', async e => {
  if (e.ctrlKey && e.shiftKey && e.key === 'C') {
    const res = await fetch('/clipboard');
    const { text } = await res.json();
    await navigator.clipboard.writeText(text);
    showToast('📋 Copied: ' + text.slice(0,30));
  }
});

// ── Fullscreen ─────────────────────────────────────────────────────────────
document.getElementById('btn-fullscreen').onclick = () => {
  const el = document.getElementById('video-container');
  document.fullscreenElement ? document.exitFullscreen() : el.requestFullscreen();
};

// ── Stats ──────────────────────────────────────────────────────────────────
function startStats() {
  setInterval(async () => {
    if (!pc || statsEl.style.display === 'none') return;
    const stats = await pc.getStats();
    let vfps = 0, vkbps = 0, akbps = 0;
    stats.forEach(s => {
      if (s.type === 'inbound-rtp') {
        if (s.kind === 'video') { vfps = s.framesPerSecond||0; vkbps = Math.round((s.bytesReceived||0)*8/1000); }
        if (s.kind === 'audio')  { akbps = Math.round((s.bytesReceived||0)*8/1000); }
      }
    });
    statsEl.textContent = `Video: ${vfps}fps ${vkbps}kb/s | Audio: ${akbps}kb/s`;
  }, 1000);
}
document.addEventListener('keydown', e => {
  if (e.shiftKey && e.key === 'S')
    statsEl.style.display = statsEl.style.display === 'none' ? 'block' : 'none';
});

btnConn.onclick = connect;
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNALING SERVER
# ═══════════════════════════════════════════════════════════════════════════════
async def index(request):
    return web.Response(text=HTML, content_type="text/html")


async def offer(request):
    params    = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    video_track = NvencCaptureTrack()
    audio_track = PulseAudioTrack()

    pc.addTrack(video_track)
    pc.addTrack(audio_track)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        async def on_message(msg):
            data = json.loads(msg)
            # Volume control
            if data.get("type") == "volume":
                audio_track.set_volume(data.get("value", 1.0))
            else:
                await handle_input(data)

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed"):
            video_track.stop()
            audio_track.stop()
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({ "sdp": pc.localDescription.sdp, "type": pc.localDescription.type })
    )


async def clipboard_get(request):
    try:
        proc = await asyncio.create_subprocess_exec(
            "xclip", "-selection", "clipboard", "-o",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        text = stdout.decode("utf-8", errors="replace")
    except:
        text = ""
    return web.Response(content_type="application/json", text=json.dumps({"text": text}))


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
SCREEN_W_REAL = int(os.environ.get("SCREEN_W", "1920"))
SCREEN_H_REAL = int(os.environ.get("SCREEN_H", "1080"))
MOUSE_MAP     = { 0: "1", 1: "2", 2: "3" }

async def xdotool(*args):
    env = {**os.environ, "DISPLAY": DISPLAY}
    proc = await asyncio.create_subprocess_exec(
        "xdotool", *args, env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

async def handle_input(data):
    t = data.get("type")
    if t == "mousemove":
        x, y = int(data["x"]*SCREEN_W_REAL), int(data["y"]*SCREEN_H_REAL)
        await xdotool("mousemove", str(x), str(y))
    elif t == "mousedown":
        x, y = int(data["x"]*SCREEN_W_REAL), int(data["y"]*SCREEN_H_REAL)
        await xdotool("mousemove", str(x), str(y))
        await xdotool("mousedown", MOUSE_MAP.get(data.get("button",0), "1"))
    elif t == "mouseup":
        await xdotool("mouseup", MOUSE_MAP.get(data.get("button",0), "1"))
    elif t == "wheel":
        await xdotool("click", "5" if data.get("dy",0) > 0 else "4")
    elif t in ("keydown", "keyup"):
        await xdotool(t.replace("down","keydown").replace("up","keyup"), data.get("key",""))
    elif t == "clipboard":
        proc = await asyncio.create_subprocess_exec(
            "xclip", "-selection", "clipboard",
            stdin=asyncio.subprocess.PIPE,
            env={**os.environ, "DISPLAY": DISPLAY},
        )
        await proc.communicate(input=data.get("text","").encode())


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()


def create_app():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_get("/clipboard", clipboard_get)
    app.on_shutdown.append(on_shutdown)
    return app


INSTALL_SCRIPT = """
apt-get install -y -qq \
    xvfb x11-xserver-utils \
    xfce4 xfce4-terminal \
    dbus-x11 \
    pulseaudio alsa-utils \
    xdotool xclip ffmpeg \
    wget 2>/dev/null
pip install -q aiortc aiohttp av numpy
"""

def setup_display():
    """Khởi động Xvfb + Xfce4 + PulseAudio"""
    print("[i] Khởi động màn hình ảo...")

    # Kill cũ nếu có
    subprocess.run(["pkill", "-f", "Xvfb"], capture_output=True)
    subprocess.run(["pkill", "-f", "xfce4-session"], capture_output=True)
    time.sleep(1)

    # Xvfb
    subprocess.Popen([
        "Xvfb", ":99", "-screen", "0", "1920x1080x24",
        "-ac", "+extension", "GLX", "+extension", "RANDR",
        "-dpi", "96", "-noreset"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    env = {**os.environ, "DISPLAY": ":99"}

    # Xfce4
    subprocess.Popen(
        ["startxfce4"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(4)

    # PulseAudio
    subprocess.run(
        ["pulseaudio", "--start", "--exit-idle-time=-1"],
        env=env, capture_output=True
    )
    time.sleep(1)

    print("[✓] Màn hình ảo đã khởi động (DISPLAY=:99)")


def setup_tunnel():
    """Tạo Cloudflare Tunnel và in link"""
    print("[i] Tạo Cloudflare Tunnel...")

    # Tải cloudflared nếu chưa có
    if not os.path.exists("/tmp/cloudflared"):
        subprocess.run([
            "wget", "-q",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-O", "/tmp/cloudflared"
        ])
        subprocess.run(["chmod", "+x", "/tmp/cloudflared"])

    def run_tunnel():
        proc = subprocess.Popen(
            ["/tmp/cloudflared", "tunnel", "--url", "http://localhost:8080"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for line in proc.stdout:
            line = line.decode("utf-8", errors="replace").strip()
            if "trycloudflare.com" in line:
                # Extract URL
                for word in line.split():
                    if "trycloudflare.com" in word:
                        print(f"\n{'='*50}")
                        print(f"  🔗 Mở link này trên browser:")
                        print(f"  {word}")
                        print(f"{'='*50}\n")
                        break

    t = threading.Thread(target=run_tunnel, daemon=True)
    t.start()
    time.sleep(5)


if __name__ == "__main__":
    # 1. Cài dependencies
    try:
        import aiortc, aiohttp, av, numpy
    except ImportError:
        print("[i] Cài dependencies...")
        subprocess.run(INSTALL_SCRIPT, shell=True)
        # Reload
        import importlib, site
        importlib.invalidate_caches()

    from aiohttp import web
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
    from av import VideoFrame, AudioFrame
    import numpy as np

    PORT = int(os.environ.get("PORT", 8080))
    HOST = os.environ.get("HOST", "0.0.0.0")
    has_gpu = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0

    # 2. Khởi động màn hình
    setup_display()

    # 3. Tạo tunnel
    setup_tunnel()

    print(f"""
╔══════════════════════════════════════════╗
║   WebRTC Remote Desktop - Bước 2        ║
║   Video + Audio (PulseAudio)            ║
║   GPU: {'✅ NVENC' if has_gpu else '❌ CPU fallback'}                    
║   http://{HOST}:{PORT}                 ║
║   Shift+S: stats                        ║
╚══════════════════════════════════════════╝
""")

    app = create_app()
    web.run_app(app, host=HOST, port=PORT)
