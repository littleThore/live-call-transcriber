"""Live call transcriber: mic ("Me") + WASAPI loopback of an output device ("Call").

Transcribes locally with faster-whisper (NVIDIA GPU: large-v3-turbo float16; otherwise CPU int8,
smaller model) in silence-split chunks and appends timestamped lines to
transcripts/call-YYYY-MM-DD-HHMM.md. Stop with Ctrl+C.

Normally started by LiveCallTranscriber.exe, which sets up its own Python environment.
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import collections
import datetime as dt
import math
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import pyaudiowpatch as pyaudio

DEBUG = False
SR = 16000
FRAME = 480  # 30 ms at 16 kHz
APP_DIR = Path(__file__).resolve().parent


def add_cuda_dll_dirs():
    """Make the pip 'nvidia-*-cu12' DLLs (cuBLAS, cuDNN) visible to ctranslate2. Returns True if found."""
    found = False
    for sp in sys.path:
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for b in base.glob("*/bin"):
            found = True
            os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(b))
            except Exception:
                pass
    if not found:
        try:  # environments with CUDA torch ship the same DLLs in torch/lib
            import torch  # noqa: F401
            found = True
        except Exception:
            pass
    return found


def load_model(name, device):
    """Load Whisper on CUDA if it really works (warm-up decode), else fall back to CPU."""
    from faster_whisper import WhisperModel
    if device in ("auto", "cuda"):
        try:
            add_cuda_dll_dirs()
            m = WhisperModel(name or "large-v3-turbo", device="cuda", compute_type="float16")
            m.detect_language(np.zeros(SR, dtype=np.float32))  # runs the encoder: fails here if cuBLAS is missing
            return m, "CUDA float16", name or "large-v3-turbo"
        except Exception as e:
            if device == "cuda":
                raise
            log(f"  (GPU not usable: {str(e).splitlines()[0][:120]}; using CPU)")
    cpu_model = name or "small"
    return WhisperModel(cpu_model, device="cpu", compute_type="int8"), "CPU int8", cpu_model

# Typical Whisper hallucinations on silence/noise (German subtitle credits etc.)
HALLUCINATIONS = [
    r"untertitel", r"amara\.org", r"zdf", r"vielen dank (fürs|für's|für das) zu(schauen|sehen)",
    r"thanks? (you )?for watching", r"bis zum nächsten mal", r"^\W*(danke|thank you)\.?\W*$",
    r"swr", r"copyright", r"^\W*$",
]
HALLU_RE = re.compile("|".join(HALLUCINATIONS), re.IGNORECASE)


def log(msg):
    print(msg, flush=True)


def collapse_repeats(text, max_n=12):
    """Cut a phrase that repeats back-to-back 3+ times down to one copy (Whisper repetition loops)."""
    words = text.split()
    out, i = [], 0
    while i < len(words):
        for n in range(max_n, 0, -1):
            chunk = words[i:i + n]
            if len(chunk) < n:
                continue
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == chunk:
                reps += 1
            if reps >= 3:
                out.extend(chunk)
                i += reps * n
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


class Capture:
    """One PyAudio input stream -> 16 kHz mono float32 frames on a queue."""

    def __init__(self, pa, label, dev, recording=True):
        self.label = label
        self.dev = dev
        self.rate = int(dev["defaultSampleRate"])
        self.ch = max(1, min(2, int(dev["maxInputChannels"])))
        g = math.gcd(SR, self.rate)
        self.up, self.down = SR // g, self.rate // g
        self.q = queue.Queue()
        self.recording = recording  # False = only meter the level (UI preview), nothing is queued
        self.peak = 0.0
        self.stream = pa.open(
            format=pyaudio.paInt16, channels=self.ch, rate=self.rate, input=True,
            input_device_index=dev["index"], frames_per_buffer=int(self.rate * 0.05),
            stream_callback=self._cb,
        )

    def _cb(self, data, frames, info, status):
        x = np.frombuffer(data, dtype=np.int16)
        if len(x):
            self.peak = max(self.peak, float(np.abs(x).max()) / 32768.0)
        if self.recording:
            self.q.put(data)
        return (None, pyaudio.paContinue)

    def take_peak(self):
        p, self.peak = self.peak, 0.0
        return p

    def to16k(self, data):
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if self.ch > 1:
            x = x.reshape(-1, self.ch).mean(axis=1)
        if self.up != self.down:
            x = resample_poly(x, self.up, self.down).astype(np.float32)
        return x

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


class Segmenter(threading.Thread):
    """Energy-based silence splitter. Emits (label, start_wallclock, audio16k) chunks."""

    def __init__(self, cap, out_q, stop, silence_s, max_s, min_speech_s):
        super().__init__(daemon=True)
        self.cap, self.out_q, self.stop = cap, out_q, stop
        self.silence_frames = int(silence_s * SR / FRAME)
        self.max_samples = int(max_s * SR)
        self.min_speech_frames = int(min_speech_s * SR / FRAME)
        self.preroll = collections.deque(maxlen=int(0.3 * SR / FRAME))
        self.floor = 0.003
        self.buf = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self):
        self.speech = []
        self.in_speech = False
        self.silent_run = 0
        self.voiced = 0
        self.t0 = None
        self.last_data = time.time()

    def emit(self):
        if self.in_speech and self.voiced >= self.min_speech_frames:
            keep = len(self.speech) - max(0, self.silent_run - int(0.25 * SR / FRAME))
            audio = np.concatenate(self.speech[:max(keep, 1)])
            if DEBUG:
                log(f"[debug] {self.cap.label} chunk {len(audio) / SR:.1f}s")
            self.out_q.put((self.cap.label, self.t0, audio))
        self.reset()

    def frame(self, f):
        rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
        thr = max(0.006, self.floor * 2.5)
        loud = rms > thr
        if not self.in_speech:
            self.floor = 0.98 * self.floor + 0.02 * min(rms, 0.05)
            self.preroll.append(f)
            if loud:
                self.in_speech = True
                self.speech = list(self.preroll)
                self.t0 = time.time() - len(self.speech) * FRAME / SR
                self.voiced, self.silent_run = 1, 0
            return
        self.speech.append(f)
        if loud:
            self.voiced += 1
            self.silent_run = 0
        else:
            self.silent_run += 1
        if self.silent_run >= self.silence_frames or len(self.speech) * FRAME >= self.max_samples:
            self.emit()

    def run(self):
        while not self.stop.is_set():
            try:
                data = self.cap.q.get(timeout=0.2)
            except queue.Empty:
                # WASAPI loopback delivers nothing while the device is silent: count wall time as silence.
                if self.in_speech and time.time() - self.last_data > self.silence_frames * FRAME / SR:
                    self.emit()
                continue
            self.last_data = time.time()
            self.buf = np.concatenate([self.buf, self.cap.to16k(data)])
            if DEBUG and time.time() - getattr(self, "_dbg", 0) > 2:
                self._dbg = time.time()
                log(f"[debug] {self.cap.label} rms={np.sqrt(np.mean(self.buf ** 2)) if len(self.buf) else 0:.4f} "
                    f"floor={self.floor:.4f} speech={self.in_speech} q={self.cap.q.qsize()}")
            n = len(self.buf) // FRAME
            for i in range(n):
                self.frame(self.buf[i * FRAME:(i + 1) * FRAME])
            self.buf = self.buf[n * FRAME:]
        self.emit()  # flush on stop


SPEAKER_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                     "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx")


class Speakers:
    """Online speaker clustering for the Call stream: voice embedding per segment, nearest centroid."""

    def __init__(self, threshold):
        import sherpa_onnx
        path = APP_DIR / "models" / SPEAKER_MODEL_URL.rsplit("/", 1)[1]
        if not path.exists():
            import urllib.request
            path.parent.mkdir(parents=True, exist_ok=True)
            log("  Downloading speaker model (26 MB) ...")
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(SPEAKER_MODEL_URL, tmp)
            tmp.replace(path)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=2)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.thr = threshold
        self.sums = []  # running sum of unit embeddings per speaker

    def embed(self, audio):
        st = self.ex.create_stream()
        st.accept_waveform(SR, audio)
        st.input_finished()
        if not self.ex.is_ready(st):
            return None
        e = np.asarray(self.ex.compute(st), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def assign(self, audio):
        """Return a 1-based speaker number, or None if the audio is too short to judge."""
        dur = len(audio) / SR
        e = self.embed(audio) if dur >= 0.8 else None
        if e is None:
            return None
        if self.sums:
            sims = [float(e @ (s / np.linalg.norm(s))) for s in self.sums]
            i = int(np.argmax(sims))
            if DEBUG:
                log(f"[debug] speaker sims {[round(x, 2) for x in sims]} ({dur:.1f}s)")
            if sims[i] >= self.thr or dur < 1.5:  # short clips are too noisy to found a new speaker
                if dur >= 1.5:
                    self.sums[i] = self.sums[i] + e
                return i + 1
        self.sums.append(e.copy())
        return len(self.sums)


class Transcriber(threading.Thread):
    def __init__(self, model, in_q, out_path, lang, langs, idle, speakers=None, on_line=None):
        super().__init__(daemon=True)
        self.model, self.in_q, self.out_path = model, in_q, out_path
        self.lang, self.langs, self.idle = lang, langs, idle
        self.speakers = speakers
        self.last_spk = None
        self.on_line = on_line  # UI hook: on_line(stamp, who, lang, text, seconds_behind)

    def pick_lang(self, audio):
        if self.lang:
            return self.lang
        try:
            lang, prob, allp = self.model.detect_language(audio)
        except Exception:
            return None
        if lang in self.langs:
            return lang
        cand = [(p, l) for l, p in allp if l in self.langs]
        return max(cand)[1] if cand else lang

    def run(self):
        while True:
            item = self.in_q.get()
            if item is None:
                break
            self.idle.clear()
            try:
                self.handle(*item)
            except Exception as e:
                log(f"[transcribe error] {e!r}")
            finally:
                if self.in_q.empty():
                    self.idle.set()

    def handle(self, label, t0, audio):
        lang = self.pick_lang(audio)
        backlog = self.in_q.qsize()
        behind = time.time() - (t0 + len(audio) / SR)
        segs, info = self.model.transcribe(
            audio, language=lang, beam_size=5 if backlog == 0 else 1,  # greedy when falling behind
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False, temperature=0.0,
        )
        lines = []  # [who, start_s, text]
        for s in segs:
            t = s.text.strip()
            if not t:
                continue
            if s.no_speech_prob > 0.6 and s.avg_logprob < -0.8:
                continue
            if s.compression_ratio > 2.4:  # Whisper stuck in a loop ("... math ... math ... math")
                t = collapse_repeats(t)
                if not t:
                    continue
            if HALLU_RE.search(t):
                continue
            who = label
            if self.speakers is not None and label == "Call":
                n = None
                try:
                    n = self.speakers.assign(audio[int(s.start * SR):int(s.end * SR)])
                except Exception as e:
                    log(f"[speaker error] {e!r}")
                if n is None:
                    n = self.last_spk  # too short to judge: most likely whoever spoke last
                self.last_spk = n or self.last_spk
                who = f"Speaker {n}" if n else "Call"
            if lines and lines[-1][0] == who:
                lines[-1][2] += " " + t
            else:
                lines.append([who, s.start, t])
        lang_tag = (lang or "?").upper()
        lag = f"   [{behind:.0f}s behind]" if behind > 15 else ""
        for who, start, text in lines:
            stamp = dt.datetime.fromtimestamp(t0 + start).strftime("%H:%M:%S")
            if self.on_line:
                self.on_line(stamp, who, lang_tag, text, behind)
            else:
                print(f"[{stamp}] {who} ({lang_tag}): {text}{lag}", flush=True)
            with open(self.out_path, "a", encoding="utf-8") as fh:
                fh.write(f"- `{stamp}` **{who}** ({lang_tag}): {text}\n")


def find_device(pa, devices, needle):
    needle = needle.lower()
    for d in devices:
        if needle in d["name"].lower():
            return d
    raise SystemExit(f"No device matching '{needle}'. Run with --list-devices.")


def main():
    ap = argparse.ArgumentParser(description="Live transcript of mic (Me) + output loopback (Call).")
    ap.add_argument("--call-device", help="substring of the output device to loop back (default: Windows default output)")
    ap.add_argument("--mic-device", help="substring of the input device (default: Windows default input)")
    ap.add_argument("--no-mic", action="store_true")
    ap.add_argument("--no-call", action="store_true")
    ap.add_argument("--lang", help="force a language (de/en); default auto, limited to --langs")
    ap.add_argument("--langs", default="de,en", help="allowed auto languages (default de,en)")
    ap.add_argument("--model", help="Whisper model (default: large-v3-turbo on GPU, small on CPU)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out-dir", help="folder for transcripts (default: ./transcripts next to the script)")
    ap.add_argument("--silence", type=float, default=0.6, help="seconds of silence that end a chunk")
    ap.add_argument("--max-chunk", type=float, default=12.0, help="max chunk length in seconds")
    ap.add_argument("--out", help="transcript path (default transcripts/discord-YYYY-MM-DD-HHMM.md)")
    ap.add_argument("--duration", type=float, help="stop after N seconds (testing)")
    ap.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)  # no-op, kept for old command lines
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-speakers", action="store_true", help="label all call audio as 'Call' (no voice separation)")
    ap.add_argument("--speaker-threshold", type=float, default=0.6,
                    help="voice similarity needed to count as the same speaker (lower = fewer speakers)")
    ap.add_argument("--no-send-menu", action="store_true", help="don't offer to send/copy the transcript at the end")
    ap.add_argument("--send", metavar="TRANSCRIPT", help="don't record: open the send/copy menu for an existing transcript")
    ap.add_argument("--debug", action="store_true", help="print levels and chunk sizes")
    ap.add_argument("--gui", action="store_true", help="open the window (default when the exe is double-clicked)")
    a = ap.parse_args()
    global DEBUG
    DEBUG = a.debug
    if a.gui:
        import gui
        gui.run(a)
        return
    if a.send:
        share_menu(a.send)
        return

    pa = pyaudio.PyAudio()
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    loops = list(pa.get_loopback_device_info_generator())
    inputs = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    inputs = [d for d in inputs if d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0
              and not d.get("isLoopbackDevice")]

    if a.list_devices:
        log("Loopback (use with --call-device):")
        for d in loops:
            log(f"  {d['name']}")
        log("Inputs (use with --mic-device):")
        for d in inputs:
            log(f"  {d['name']}")
        return

    log("")

    call_dev = mic_dev = None
    if not a.no_call:
        call_dev = find_device(pa, loops, a.call_device) if a.call_device else pa.get_default_wasapi_loopback()
    if not a.no_mic:
        if a.mic_device:
            mic_dev = find_device(pa, inputs, a.mic_device)
        else:
            mic_dev = pa.get_device_info_by_index(wasapi["defaultInputDevice"])

    log("Loading Whisper (first start downloads the model, ~0.5-1.6 GB) ...")
    model, how, model_name = load_model(a.model, a.device)
    log(f"  Model: {model_name} on {how}")

    out_dir = Path(a.out_dir) if a.out_dir else APP_DIR / "transcripts"
    out = Path(a.out) if a.out else out_dir / f"call-{dt.datetime.now():%Y-%m-%d-%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", encoding="utf-8") as fh:
        if new:
            fh.write(f"# Call transcript {dt.datetime.now():%Y-%m-%d %H:%M}\n\n")
        fh.write(f"_Session started {dt.datetime.now():%H:%M:%S}. Me = "
                 f"{mic_dev['name'] if mic_dev else 'off'}; Call = {call_dev['name'] if call_dev else 'off'}._\n\n")

    chunks = queue.Queue()
    stop = threading.Event()
    idle = threading.Event()
    idle.set()
    speakers = None
    if call_dev is not None and not a.no_speakers:
        try:
            speakers = Speakers(a.speaker_threshold)
            log("  Speaker separation on (call voices -> Speaker 1, 2, ...)")
        except Exception as e:
            log(f"  (Speaker separation off: {e})")
    worker = Transcriber(model, chunks, out, a.lang, [s.strip() for s in a.langs.split(",") if s.strip()], idle,
                         speakers)
    worker.start()

    caps, segs = [], []
    for label, dev in (("Me", mic_dev), ("Call", call_dev)):
        if dev is None:
            continue
        cap = Capture(pa, label, dev)
        caps.append(cap)
        seg = Segmenter(cap, chunks, stop, a.silence, a.max_chunk, 0.3)
        seg.start()
        segs.append(seg)
        log(f"  {label:<4} <- {dev['name']} ({cap.rate} Hz, {cap.ch} ch)")

    log(f"Transcribing live -> {out}")
    log("Press Ctrl+C to stop.\n")
    t_start = time.time()
    try:
        while True:
            time.sleep(0.25)
            if a.duration and time.time() - t_start >= a.duration:
                break
    except KeyboardInterrupt:
        pass

    for c in caps:
        c.close()
    stop.set()
    for s in segs:
        s.join(timeout=2)
    log(f"\nStopping ... finishing the last {chunks.qsize()} chunk(s). Press Ctrl+C again to skip them.")
    chunks.put(None)
    try:
        while worker.is_alive():
            worker.join(timeout=0.5)
    except KeyboardInterrupt:
        log(f"Skipped {max(chunks.qsize() - 1, 0)} untranscribed chunk(s).")
    pa.terminate()
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(f"\n_Session ended {dt.datetime.now():%H:%M:%S}._\n\n")
    log(f"Saved: {out}")
    if not a.no_send_menu:
        try:
            share_menu(out)
        except (KeyboardInterrupt, EOFError):
            pass


# ---------------------------------------------------------------- sharing the transcript

SEND_PROMPT = ("Here is the transcript of a voice call I just had (live speech-to-text, so names and words "
               "may be garbled; 'Me' is me, 'Speaker N'/'Call' are the others). Keep it as context for this "
               "chat and reply with a short list of the useful points.\n\n")


def claude_exe():
    import shutil
    exe = shutil.which("claude")
    if exe:
        return exe
    p = Path.home() / ".local" / "bin" / "claude.exe"
    return str(p) if p.exists() else None


def recent_claude_sessions(n=9):
    """Most recently used Claude Code sessions: id, cwd, title (from ~/.claude/projects/*/*.jsonl)."""
    import json
    root = Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return []
    files = sorted(root.glob("*/*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)[: n * 3]
    found = []
    for f in files:
        size = f.stat().st_size
        with open(f, "rb") as fh:
            head = fh.read(300_000)
            fh.seek(max(0, size - 300_000))
            tail = fh.read()
        title = cwd = first = None
        for raw in head.splitlines() + tail.splitlines():
            if b'"customTitle"' in raw or (cwd is None and b'"cwd"' in raw) or (first is None and b'"type":"user"' in raw):
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                if rec.get("customTitle"):
                    title = rec["customTitle"]  # later records win: the latest rename
                if cwd is None and rec.get("cwd"):
                    cwd = rec["cwd"]
                if first is None and rec.get("type") == "user":
                    c = (rec.get("message") or {}).get("content")
                    if isinstance(c, list):
                        c = next((x.get("text") for x in c if isinstance(x, dict) and x.get("type") == "text"), None)
                    if isinstance(c, str) and c.strip() and not c.lstrip().startswith("<"):
                        first = " ".join(c.split())
        if not cwd:
            continue
        found.append({"id": f.stem, "cwd": cwd, "title": title or (first or "(untitled)")[:70],
                      "when": dt.datetime.fromtimestamp(f.stat().st_mtime)})
        if len(found) >= n:
            break
    return found


def run_claude(args, prompt, cwd):
    import subprocess
    exe = claude_exe()
    r = subprocess.run([exe, "-p", *args], input=prompt, text=True, encoding="utf-8", errors="replace",
                       capture_output=True, cwd=cwd if cwd and Path(cwd).is_dir() else None, timeout=900)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[:500])
    return r.stdout


def copy_to_clipboard(text):
    import ctypes
    from ctypes import wintypes
    k, u = ctypes.windll.kernel32, ctypes.windll.user32
    k.GlobalAlloc.restype = wintypes.HGLOBAL
    k.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    k.GlobalLock.restype = ctypes.c_void_p
    k.GlobalLock.argtypes = [wintypes.HGLOBAL]
    k.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    u.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    u.SetClipboardData.restype = wintypes.HANDLE
    data = text.encode("utf-16-le") + b"\0\0"
    h = k.GlobalAlloc(0x0002, len(data))  # GMEM_MOVEABLE
    p = k.GlobalLock(h)
    ctypes.memmove(p, data, len(data))
    k.GlobalUnlock(h)
    for _ in range(10):
        if u.OpenClipboard(None):
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("clipboard is busy")
    try:
        u.EmptyClipboard()
        u.SetClipboardData(13, h)  # CF_UNICODETEXT
    finally:
        u.CloseClipboard()


def share_menu(path):
    text = Path(path).read_text(encoding="utf-8")
    if "**" not in text:
        return  # nothing was transcribed
    sessions = recent_claude_sessions() if claude_exe() else []
    log("\nWhat should happen with the transcript?")
    for i, s in enumerate(sessions, 1):
        log(f"  [{i}] send to Claude Code chat: {s['title']}  ({Path(s['cwd']).name}, {s['when']:%d.%m. %H:%M})")
    if claude_exe():
        log("  [n] start a new Claude Code chat with it")
    log("  [c] copy to clipboard (paste into any chat)")
    log("  [Enter] nothing, just keep the file")
    if sessions or claude_exe():
        log("  (Sending uploads the transcript to Claude/Anthropic.)")
    choice = input("> ").strip().lower()
    if not choice:
        return
    if choice == "c":
        copy_to_clipboard(text)
        log("Copied.")
        return
    if choice == "n" and claude_exe():
        import json
        log("Sending to a new Claude Code chat ...")
        res = json.loads(run_claude(["--output-format", "json"], SEND_PROMPT + text, os.getcwd()))
        log("\n" + res.get("result", "").strip())
        log(f"\nOpen this chat later with:  claude --resume {res.get('session_id')}")
        return
    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
        s = sessions[int(choice) - 1]
        log(f"Sending to '{s['title']}' ... (if that chat is open in the Claude app, reopen it to see the message)")
        log("\n" + run_claude(["--resume", s["id"]], SEND_PROMPT + text, s["cwd"]).strip())
        return
    log("Unknown choice, transcript kept as file only.")


if __name__ == "__main__":
    main()
