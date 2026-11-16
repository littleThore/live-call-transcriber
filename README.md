# Live Call Transcriber

Writes down a voice call (Discord, Teams, anything) live, on your own PC. Nothing is uploaded.

- **Me**: your microphone
- **Call**: everything your PC plays (WASAPI loopback of an output device)

Speech is split at pauses, transcribed locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), printed to the console and appended to a Markdown file:

```
- `22:21:56` **Call** (DE): Hallo zusammen, das ist ein kurzer Test der Live-Transkription.
- `22:22:05` **Me** (EN): and this part is in English.
```

German and English are detected automatically per chunk.

**Speaker separation:** call voices are told apart by a voice fingerprint (a [WeSpeaker](https://github.com/wenet-e2e/wespeaker) speaker embedding via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)) and labelled `Speaker 1`, `Speaker 2`, … live. Numbers are per session; the tool doesn't know names. If one person gets split into two speakers, lower `--speaker-threshold` (e.g. 0.5). If two people get merged, raise it (e.g. 0.7).

**After the call** (Ctrl+C), you can:
- send the transcript into one of your recent **Claude Code** chats (picked from a list, via `claude -p --resume`),
- start a **new Claude Code chat** with it,
- or **copy it to the clipboard** to paste anywhere (ChatGPT, Discord, Notion, …).

Sending uploads the transcript to Anthropic. To open the menu again later for an old transcript: `LiveCallTranscriber.exe --send transcripts\call-….md`.

Note: in many countries (e.g. Germany, §201 StGB) transcribing others without their consent is illegal; you are responsible for how you use this.

## Download and run (Windows 10/11, 64-bit)

1. Download `LiveCallTranscriber.exe` from [Releases](../../releases/latest).
2. Put it in its own folder and double-click it. Transcripts are saved to a `transcripts` folder next to the exe.
3. Stop with **Ctrl+C**.

The first start sets everything up by itself in `%LOCALAPPDATA%\LiveCallTranscriber` and takes a few minutes:

| PC | Download | Model | Speed |
|---|---|---|---|
| NVIDIA GPU | about 1.5 GB of CUDA libraries, plus a 1.6 GB model | `large-v3-turbo` | real time, best quality |
| No NVIDIA GPU | about 0.3 GB, plus a 0.5 GB model | `small` (CPU) | slower, lower quality |

Windows SmartScreen may warn because the exe is unsigned: click **More info → Run anyway**. The source is in this repo, and you can build it yourself (see below).

## Picking the right devices

The **Call** stream records the Windows default output by default. If Discord plays through a different device (Discord → Settings → Voice & Video → Output Device), point the tool at that device:

```
LiveCallTranscriber.exe --list-devices
LiveCallTranscriber.exe --call-device "A50 Voice"
```

To save your options for double-clicking, create `default-args.txt` next to the exe (see `default-args.example.txt`). It is only used when the exe is started without arguments.

| Option | |
|---|---|
| `--call-device NAME` | part of the output device name to record |
| `--mic-device NAME` | part of the microphone name |
| `--no-mic` / `--no-call` | record only one side |
| `--lang de` | force one language (default: auto, limited to `--langs de,en`) |
| `--model NAME` | Whisper model, e.g. `medium`, `large-v3` |
| `--device cpu` | force CPU |
| `--out-dir PATH` | where transcripts go |
| `--silence 0.6` / `--max-chunk 12` | chunking: pause length that ends a chunk, max chunk length (s) |
| `--no-speakers` | don't separate call voices (everything is `Call`) |
| `--speaker-threshold 0.6` | voice similarity needed to count as the same speaker |
| `--send FILE` | don't record, open the send/copy menu for an existing transcript |
| `--no-send-menu` | skip the menu at the end |
| `--debug` | show audio levels and chunk sizes |

The send menu needs the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude`) on PATH or in `%USERPROFILE%\.local\bin`; without it, only the clipboard option is shown.

**Lagging behind?** Whisper needs about 2 GB of GPU memory. If games or another GPU app fill the VRAM, transcription slows down a lot. The console then shows `[Ns behind]` and the tool switches to faster decoding. Close other GPU-heavy apps, or use `--model medium`.

Headphones work best. With speakers, your mic also picks up the call, so call lines show up twice.

## Build it yourself

```
build.cmd
```

This uses the C# compiler that ships with Windows and writes `dist\LiveCallTranscriber.exe`. The launcher (`launcher/Launcher.cs`) embeds `live_transcribe.py` and, on first start, downloads [uv](https://github.com/astral-sh/uv), creates a Python 3.11 environment and installs `faster-whisper`, `pyaudiowpatch`, `scipy` and `numpy` (plus `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` on NVIDIA PCs).

To run the script directly in your own Python 3.11 environment instead:

```
pip install -r requirements.txt
python live_transcribe.py
```
