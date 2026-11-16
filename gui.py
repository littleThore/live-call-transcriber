"""Window for Live Call Transcriber: pick devices (with live level meters), start/stop, live transcript,
send to Claude Code or copy. Started with `live_transcribe.py --gui` (the exe does this on double-click)."""
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

import live_transcribe as lt

BG = "#15171c"
PANEL = "#1d2027"
FIELD = "#262a33"
LINE = "#2f3440"
FG = "#e7e9ee"
MUTED = "#8a91a0"
ACCENT = "#8fa8ff"
REC = "#ff5c5c"
METER_OK = "#5fd38a"
METER_HOT = "#f2b84b"
SPEAKER_COLORS = ["#7cc4ff", "#ffb86b", "#c6a8ff", "#7ee2a8", "#ff8fb4", "#f3d66d", "#6fe0e0", "#ff9e7a"]
OFF = "Off"
LANGS = {"Auto (Deutsch / English)": None, "Deutsch": "de", "English": "en"}
SETTINGS = lt.APP_DIR / "settings.json"


def strip_loop(name):
    return name.replace(" [Loopback]", "")


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.ui_q = queue.Queue()
        self.model = self.speakers = None
        self.model_info = ""
        self.caps = {"Call": None, "Me": None}
        self.recording = self.finishing = False
        self.out_path = None
        self.lines_written = 0
        self.speaker_tags = {}
        self.out_dir = Path(args.out_dir) if args.out_dir else lt.APP_DIR / "transcripts"
        self.settings = self.load_settings()
        self.pa = lt.pyaudio.PyAudio()
        self.claude = lt.claude_exe()
        self.sessions = []

        self.build()
        self.load_devices(initial=True)
        threading.Thread(target=self.load_model, daemon=True).start()
        self.root.after(60, self.tick)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ layout
    def build(self):
        r = self.root
        r.title("Live Call Transcriber")
        r.configure(bg=BG)
        r.minsize(620, 520)
        r.geometry(self.settings.get("geometry", "760x640"))

        st = ttk.Style(r)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10), bordercolor=LINE,
                     lightcolor=BG, darkcolor=BG, troughcolor=FIELD, focuscolor=ACCENT)
        st.configure("Panel.TFrame", background=PANEL)
        st.configure("TFrame", background=BG)
        st.configure("Cap.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI Semibold", 8))
        st.configure("Panel.TLabel", background=PANEL, foreground=FG)
        st.configure("Status.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        st.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=FG, arrowcolor=MUTED,
                     bordercolor=LINE, lightcolor=FIELD, darkcolor=FIELD, padding=(8, 5))
        st.map("TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
               foreground=[("disabled", MUTED)], selectbackground=[("readonly", FIELD)],
               selectforeground=[("readonly", FG)], arrowcolor=[("disabled", LINE)])
        r.option_add("*TCombobox*Listbox.background", FIELD)
        r.option_add("*TCombobox*Listbox.foreground", FG)
        r.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        r.option_add("*TCombobox*Listbox.selectForeground", BG)
        r.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))
        st.configure("Vertical.TScrollbar", background=FIELD, troughcolor=PANEL, bordercolor=PANEL,
                     arrowcolor=MUTED, lightcolor=FIELD, darkcolor=FIELD, gripcount=0)
        st.map("Vertical.TScrollbar", background=[("active", LINE)])
        st.configure("TCheckbutton", background=PANEL, foreground=FG, indicatorbackground=FIELD,
                     indicatorforeground=ACCENT)
        st.map("TCheckbutton", background=[("active", PANEL)], indicatorbackground=[("disabled", PANEL)])
        for name, bg, fg in (("Primary", ACCENT, BG), ("Rec", REC, "#1a0d0d"), ("Ghost", FIELD, FG)):
            st.configure(f"{name}.TButton", background=bg, foreground=fg, bordercolor=bg, lightcolor=bg,
                         darkcolor=bg, padding=(14, 6), font=("Segoe UI Semibold", 10))
            st.map(f"{name}.TButton", background=[("disabled", PANEL), ("active", bg)],
                   foreground=[("disabled", MUTED)])

        top = ttk.Frame(r, style="Panel.TFrame", padding=(18, 14, 18, 14))
        top.pack(fill="x")
        top.columnconfigure(0, weight=1)

        self.combo, self.meter = {}, {}
        row = 0
        for key, caption in (("Call", "CALL AUDIO  ·  what you hear (Discord's output device)"),
                             ("Me", "YOUR MICROPHONE")):
            ttk.Label(top, text=caption, style="Cap.TLabel").grid(row=row, column=0, columnspan=2, sticky="w",
                                                                   pady=(0 if row == 0 else 10, 3))
            row += 1
            cb = ttk.Combobox(top, state="readonly")
            cb.grid(row=row, column=0, sticky="ew")
            cb.bind("<<ComboboxSelected>>", lambda e, k=key: self.on_device(k))
            m = tk.Canvas(top, width=150, height=10, bg=FIELD, highlightthickness=0, bd=0)
            m.grid(row=row, column=1, padx=(12, 0))
            self.combo[key], self.meter[key] = cb, m
            row += 1

        opts = ttk.Frame(top, style="Panel.TFrame")
        opts.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(opts, text="LANGUAGE", style="Cap.TLabel").pack(side="left")
        self.lang = ttk.Combobox(opts, state="readonly", width=22, values=list(LANGS))
        self.lang.set(self.settings.get("lang", list(LANGS)[0]) if self.settings.get("lang") in LANGS else list(LANGS)[0])
        self.lang.pack(side="left", padx=(8, 16))
        self.spk_var = tk.BooleanVar(value=self.settings.get("speakers", not self.args.no_speakers))
        self.spk_check = ttk.Checkbutton(opts, text="Separate call voices", variable=self.spk_var)
        self.spk_check.pack(side="left")
        self.refresh_btn = ttk.Button(opts, text="↻ Devices", style="Ghost.TButton", command=self.refresh_devices)
        self.refresh_btn.pack(side="right")

        bar = ttk.Frame(r, padding=(18, 12, 18, 6))
        bar.pack(fill="x")
        self.start_btn = ttk.Button(bar, text="Loading model …", style="Primary.TButton", command=self.toggle,
                                    state="disabled")
        self.start_btn.pack(side="left")
        self.status = ttk.Label(bar, text="", style="Status.TLabel")
        self.status.pack(side="left", padx=14)

        foot = ttk.Frame(r, padding=(18, 0, 18, 14))
        foot.pack(side="bottom", fill="x")  # packed before the text so it never gets squeezed out

        body = tk.Frame(r, bg=LINE, padx=1, pady=1)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        self.text = tk.Text(body, bg=PANEL, fg=FG, insertbackground=FG, relief="flat", wrap="word",
                            font=("Segoe UI", 10), padx=12, pady=10, spacing1=2, spacing3=4, state="disabled",
                            selectbackground=ACCENT, selectforeground=BG)
        sb = ttk.Scrollbar(body, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.tag_configure("time", foreground=MUTED, font=("Consolas", 9))
        self.text.tag_configure("Me", foreground=FG, font=("Segoe UI Semibold", 10))
        self.text.tag_configure("Call", foreground=MUTED, font=("Segoe UI Semibold", 10))
        self.text.tag_configure("note", foreground=MUTED, font=("Segoe UI", 9, "italic"))
        self.note("Pick the devices above. The meters move when a device carries sound; "
                  "the call meter should move when someone in Discord talks.")

        if self.claude:
            self.send_combo = ttk.Combobox(foot, state="readonly", width=44, postcommand=self.refresh_sessions)
            self.send_combo.pack(side="left")
            self.send_btn = ttk.Button(foot, text="Send to Claude Code", style="Ghost.TButton",
                                       command=self.send, state="disabled")
            self.send_btn.pack(side="left", padx=(8, 0))
            self.refresh_sessions()
        else:
            self.send_combo = self.send_btn = None
        self.copy_btn = ttk.Button(foot, text="Copy", style="Ghost.TButton", command=self.copy, state="disabled")
        self.copy_btn.pack(side="right")
        ttk.Button(foot, text="Open folder", style="Ghost.TButton", command=self.open_folder).pack(
            side="right", padx=(0, 8))

    # ------------------------------------------------------------------ devices
    def load_devices(self, initial=False):
        w = self.pa.get_host_api_info_by_type(lt.pyaudio.paWASAPI)
        self.loops = list(self.pa.get_loopback_device_info_generator())
        ins = [self.pa.get_device_info_by_index(i) for i in range(self.pa.get_device_count())]
        self.inputs = [d for d in ins if d["hostApi"] == w["index"] and d["maxInputChannels"] > 0
                       and not d.get("isLoopbackDevice")]
        self.devs = {"Call": {strip_loop(d["name"]): d for d in self.loops},
                     "Me": {d["name"]: d for d in self.inputs}}
        try:
            def_call = strip_loop(self.pa.get_default_wasapi_loopback()["name"])
        except Exception:
            def_call = OFF
        try:
            def_mic = self.pa.get_device_info_by_index(w["defaultInputDevice"])["name"]
        except Exception:
            def_mic = OFF
        wanted = {
            "Call": (self.args.call_device if initial else None, self.settings.get("call"), def_call,
                     OFF if self.args.no_call and initial else None),
            "Me": (self.args.mic_device if initial else None, self.settings.get("mic"), def_mic,
                   OFF if self.args.no_mic and initial else None),
        }
        for key in ("Call", "Me"):
            names = list(self.devs[key])
            self.combo[key]["values"] = [OFF] + names
            arg, saved, default, forced_off = wanted[key]
            pick = forced_off or None
            if not pick and arg:
                pick = next((n for n in names if arg.lower() in n.lower()), None)
            if not pick and saved and (saved == OFF or saved in names):
                pick = saved
            self.combo[key].set(pick or (default if default in names else OFF))
            self.open_preview(key)

    def open_preview(self, key):
        old = self.caps.get(key)
        if old:
            old.close()
        self.caps[key] = None
        name = self.combo[key].get()
        if name == OFF or name not in self.devs[key]:
            return
        try:
            self.caps[key] = lt.Capture(self.pa, key, self.devs[key][name], recording=False)
        except Exception as e:
            self.set_status(f"Can't open {name}: {e}")

    def on_device(self, key):
        self.open_preview(key)
        self.save_settings()

    def refresh_devices(self):
        if self.recording or self.finishing:
            return
        for k in self.caps:
            if self.caps[k]:
                self.caps[k].close()
                self.caps[k] = None
        self.save_settings()
        self.pa.terminate()
        self.pa = lt.pyaudio.PyAudio()
        self.load_devices()
        self.set_status("Device list refreshed.")

    def draw_meters(self):
        for key, m in self.meter.items():
            cap = self.caps.get(key)
            p = cap.take_peak() if cap else 0.0
            db = 20 * np.log10(max(p, 1e-5))
            frac = min(1.0, max(0.0, (db + 60) / 60))
            prev = getattr(m, "level", 0.0)
            level = frac if frac > prev else prev * 0.82 + frac * 0.18  # fast attack, slow release
            m.level = level
            m.delete("all")
            w = int(150 * level)
            if w > 0:
                m.create_rectangle(0, 0, w, 10, fill=METER_HOT if db > -6 else METER_OK, width=0)

    # ------------------------------------------------------------------ model
    def load_model(self):
        self.ui_q.put(("status", "Loading Whisper … (the first start downloads the model)"))
        try:
            model, how, name = lt.load_model(self.args.model, self.args.device)
        except Exception as e:
            self.ui_q.put(("fatal", f"Couldn't load Whisper: {e}"))
            return
        try:  # voice model first, so the very first recording already separates speakers
            self.ui_q.put(("speakers", lt.Speakers(self.args.speaker_threshold)))
        except Exception as e:
            self.ui_q.put(("status", f"Voice separation unavailable: {e}"))
        self.ui_q.put(("model", model, f"{name} · {how}"))

    # ------------------------------------------------------------------ recording
    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start()

    def start(self):
        caps = {k: c for k, c in self.caps.items() if c}
        if not caps:
            messagebox.showinfo("Live Call Transcriber", "Pick at least one device (call audio or microphone).")
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.out_dir / f"call-{dt.datetime.now():%Y-%m-%d-%H%M%S}.md"
        call = caps.get("Call")
        mic = caps.get("Me")
        with open(self.out_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Call transcript {dt.datetime.now():%Y-%m-%d %H:%M}\n\n")
            fh.write(f"_Session started {dt.datetime.now():%H:%M:%S}. Me = {mic.dev['name'] if mic else 'off'}; "
                     f"Call = {strip_loop(call.dev['name']) if call else 'off'}._\n\n")
        self.lines_written = 0
        use_spk = bool(call and self.speakers and self.spk_var.get())
        if use_spk:
            self.speakers.sums = []
            self.speaker_tags.clear()
        self.chunks = queue.Queue()
        self.stop_ev = threading.Event()
        lang = LANGS[self.lang.get()]
        self.worker = lt.Transcriber(self.model, self.chunks, self.out_path, lang, ["de", "en"], threading.Event(),
                                     self.speakers if use_spk else None,
                                     on_line=lambda *a: self.ui_q.put(("line",) + a))
        self.worker.start()
        self.segs = []
        for c in caps.values():
            c.q = queue.Queue()
            c.recording = True
            s = lt.Segmenter(c, self.chunks, self.stop_ev, self.args.silence, self.args.max_chunk, 0.3)
            s.start()
            self.segs.append(s)
        self.recording = True
        self.started_at = time.time()
        self.set_controls()
        self.set_status(f"Recording · {self.model_info}")
        self.note(f"Recording started {dt.datetime.now():%H:%M:%S}  →  {self.out_path.name}")

    def stop(self):
        self.recording = False
        self.finishing = True
        for c in self.caps.values():
            if c:
                c.recording = False
        self.stop_ev.set()
        self.set_controls()

        def finish():
            for s in self.segs:
                s.join(timeout=3)
            self.chunks.put(None)
            self.worker.join()
            with open(self.out_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n_Session ended {dt.datetime.now():%H:%M:%S}._\n")
            self.ui_q.put(("stopped",))

        threading.Thread(target=finish, daemon=True).start()

    def set_controls(self):
        busy = self.recording or self.finishing
        for cb in self.combo.values():
            cb.configure(state="disabled" if busy else "readonly")
        self.lang.configure(state="disabled" if busy else "readonly")
        self.spk_check.configure(state="disabled" if busy else "normal")
        self.refresh_btn.configure(state="disabled" if busy else "normal")
        if self.recording:
            self.start_btn.configure(text="■  Stop", style="Rec.TButton", state="normal")
        elif self.finishing:
            self.start_btn.configure(text="Finishing …", style="Primary.TButton", state="disabled")
        else:
            self.start_btn.configure(text="●  Start", style="Primary.TButton",
                                     state="normal" if self.model else "disabled")
        has = self.lines_written > 0
        self.copy_btn.configure(state="normal" if has else "disabled")
        if self.send_btn:
            self.send_btn.configure(state="normal" if has else "disabled")

    # ------------------------------------------------------------------ transcript view
    def tag_for(self, who):
        if who in ("Me", "Call"):
            return (who,)
        tag = "spk_" + who.replace(" ", "_")  # Tk would split a tag name at the space
        if tag not in self.speaker_tags:
            color = SPEAKER_COLORS[len(self.speaker_tags) % len(SPEAKER_COLORS)]
            self.text.tag_configure(tag, foreground=color, font=("Segoe UI Semibold", 10))
            self.speaker_tags[tag] = color
        return (tag,)

    def add_line(self, stamp, who, lang, text, behind):
        at_end = self.text.yview()[1] > 0.98
        self.text.configure(state="normal")
        self.text.insert("end", f"{stamp}  ", "time")
        self.text.insert("end", f"{who}  ", self.tag_for(who))
        self.text.insert("end", text + "\n")
        self.text.configure(state="disabled")
        if at_end:
            self.text.see("end")
        self.lines_written += 1
        if self.lines_written == 1:
            self.set_controls()
        if self.recording:
            self.set_status(f"Recording · {self.model_info}" + (f" · {behind:.0f} s behind" if behind > 15 else ""))

    def note(self, msg):
        self.text.configure(state="normal")
        self.text.insert("end", msg + "\n", "note")
        self.text.configure(state="disabled")
        self.text.see("end")

    def set_status(self, msg):
        self.status.configure(text=msg)

    # ------------------------------------------------------------------ event pump
    def tick(self):
        try:
            while True:
                ev = self.ui_q.get_nowait()
                kind = ev[0]
                if kind == "line":
                    self.add_line(*ev[1:])
                elif kind == "model":
                    self.model, self.model_info = ev[1], ev[2]
                    self.set_status(f"Ready · {self.model_info}")
                    self.set_controls()
                elif kind == "speakers":
                    self.speakers = ev[1]
                elif kind == "status":
                    self.set_status(ev[1])
                elif kind == "fatal":
                    self.set_status(ev[1])
                    messagebox.showerror("Live Call Transcriber", ev[1])
                elif kind == "stopped":
                    self.finishing = False
                    self.set_controls()
                    self.note(f"Saved  →  {self.out_path.name}   (Open folder)")
                    self.set_status(f"Ready · {self.model_info}")
                elif kind == "sent":
                    self.show_reply(*ev[1:])
        except queue.Empty:
            pass
        self.draw_meters()
        self.root.after(60, self.tick)

    # ------------------------------------------------------------------ sharing
    def refresh_sessions(self):
        if not self.send_combo:
            return
        try:
            self.sessions = lt.recent_claude_sessions(12)
        except Exception:
            self.sessions = []
        vals = ["New Claude Code chat"] + [f"{s['title'][:60]}  ·  {Path(s['cwd']).name}" for s in self.sessions]
        cur = self.send_combo.current()
        self.send_combo["values"] = vals
        self.send_combo.current(cur if 0 <= cur < len(vals) else 0)

    def send(self):
        if not self.out_path:
            return
        idx = self.send_combo.current()
        target = None if idx <= 0 else self.sessions[idx - 1]
        name = "a new chat" if target is None else f"'{target['title']}'"
        if not messagebox.askyesno("Send to Claude Code",
                                   f"Send this transcript to {name}?\nIt is uploaded to Claude (Anthropic)."):
            return
        text = Path(self.out_path).read_text(encoding="utf-8")
        self.send_btn.configure(state="disabled", text="Sending …")

        def work():
            try:
                if target is None:
                    res = json.loads(lt.run_claude(["--output-format", "json"], lt.SEND_PROMPT + text, str(self.out_dir)))
                    reply = res.get("result", "").strip()
                    reply += f"\n\nOpen this chat with:  claude --resume {res.get('session_id')}"
                else:
                    reply = lt.run_claude(["--resume", target["id"]], lt.SEND_PROMPT + text, target["cwd"]).strip()
                    reply += "\n\n(If that chat is open in the Claude app, reopen it to see the message.)"
                self.ui_q.put(("sent", name, reply))
            except Exception as e:
                self.ui_q.put(("sent", name, f"Sending failed: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def show_reply(self, name, reply):
        self.send_btn.configure(state="normal", text="Send to Claude Code")
        win = tk.Toplevel(self.root, bg=BG)
        win.title(f"Claude · {name}")
        win.geometry("620x420")
        t = tk.Text(win, bg=PANEL, fg=FG, relief="flat", wrap="word", font=("Segoe UI", 10), padx=14, pady=12)
        t.pack(fill="both", expand=True, padx=12, pady=12)
        t.insert("1.0", reply)
        t.configure(state="disabled")

    def copy(self):
        if self.out_path:
            lt.copy_to_clipboard(Path(self.out_path).read_text(encoding="utf-8"))
            self.set_status("Transcript copied to the clipboard.")

    def open_folder(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(self.out_dir)

    # ------------------------------------------------------------------ settings / close
    def load_settings(self):
        try:
            return json.loads(SETTINGS.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_settings(self):
        try:
            data = {"call": self.combo["Call"].get(), "mic": self.combo["Me"].get(), "lang": self.lang.get(),
                    "speakers": bool(self.spk_var.get()), "geometry": self.root.geometry()}
            SETTINGS.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except Exception:
            pass

    def on_close(self):
        if self.recording or self.finishing:
            if not messagebox.askyesno("Live Call Transcriber",
                                       "Recording is still running. Stop and close?\n"
                                       "Lines that are not transcribed yet are lost."):
                return
            if self.out_path:
                with open(self.out_path, "a", encoding="utf-8") as fh:
                    fh.write(f"\n_Session ended {dt.datetime.now():%H:%M:%S} (window closed)._\n")
        self.save_settings()
        for c in self.caps.values():
            if c:
                c.close()
        try:
            self.pa.terminate()
        except Exception:
            pass
        self.root.destroy()


def run(args):
    # pythonw has no console: keep a log next to the settings instead.
    if sys.stdout is None or sys.stderr is None:
        logf = open(lt.APP_DIR / "gui.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or logf
        sys.stderr = sys.stderr or logf
    print(f"--- GUI start {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LiveCallTranscriber")
    except Exception:
        pass
    root = tk.Tk()

    def report(exc, val, tb):
        msg = "".join(traceback.format_exception(exc, val, tb))
        print(msg, file=sys.stderr)
        messagebox.showerror("Live Call Transcriber", f"{val}\n\nDetails are in {lt.APP_DIR / 'gui.log'}")

    root.report_callback_exception = report
    App(root, args)
    dark_title_bar(root)
    root.mainloop()


def dark_title_bar(root):
    try:
        import ctypes
        root.update_idletasks()
        hwnd = int(root.wm_frame(), 16)
        on = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), ctypes.sizeof(on))
    except Exception:
        pass
