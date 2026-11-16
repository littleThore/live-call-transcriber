// LiveCallTranscriber.exe: one-file launcher.
// First start: downloads uv, creates a private Python 3.11 env in %LOCALAPPDATA%\LiveCallTranscriber
// and installs faster-whisper (+ CUDA 12 libraries when an NVIDIA GPU is present).
// Every start: writes the embedded live_transcribe.py there and runs it in this console.
// Transcripts go to a "transcripts" folder next to the exe.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Reflection;
using System.Text;

class Launcher
{
    const string UvVersion = "0.13.7";
    const string EnvVersion = "2"; // bump when the package list changes
    const string BasePackages = "faster-whisper==1.2.1 sherpa-onnx==2.0.1 pyaudiowpatch scipy numpy";
    const string CudaPackages = "nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*";

    static string home;

    static int Main(string[] args)
    {
        int code = 1;
        try { code = Run(args); }
        catch (Exception e) { Console.WriteLine(); Console.WriteLine("ERROR: " + e.Message); }
        if (!noPause && !Console.IsInputRedirected && Environment.GetEnvironmentVariable("LIVE_TRANSCRIBER_NOPAUSE") == null)
        {
            Console.WriteLine();
            Console.Write("Press any key to close ...");
            try { Console.ReadKey(true); } catch { }
        }
        return code;
    }

    static int Run(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.Title = "Live Call Transcriber";
        // Ctrl+C belongs to the Python child (it stops cleanly); the launcher just waits.
        Console.CancelKeyPress += (s, e) => { e.Cancel = true; };

        string exeDir = AppDomain.CurrentDomain.BaseDirectory;
        home = Environment.GetEnvironmentVariable("LIVE_TRANSCRIBER_HOME");
        if (string.IsNullOrEmpty(home))
            home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LiveCallTranscriber");
        Directory.CreateDirectory(home);

        string python = EnsureEnvironment();

        string script = Path.Combine(home, "live_transcribe.py");
        Assembly asm = Assembly.GetExecutingAssembly();
        foreach (string res in asm.GetManifestResourceNames())
        {
            if (!res.EndsWith(".py")) continue;
            using (Stream s = asm.GetManifestResourceStream(res))
            using (FileStream f = File.Create(Path.Combine(home, res))) s.CopyTo(f);
        }

        // Double-click (no arguments) opens the window; arguments keep the console mode.
        bool gui = args.Length == 0;
        List<string> argv = new List<string>(args);
        string defaults = Path.Combine(exeDir, "default-args.txt");
        if (argv.Count == 0 && File.Exists(defaults))
        {
            foreach (string line in File.ReadAllLines(defaults))
            {
                string t = line.Trim();
                if (t.Length > 0 && !t.StartsWith("#")) argv.AddRange(SplitArgs(t));
            }
        }
        if (!argv.Contains("--out-dir") && !argv.Contains("--out"))
        {
            argv.Add("--out-dir");
            argv.Add(Path.Combine(exeDir, "transcripts"));
        }

        if (gui) argv.Add("--gui");

        StringBuilder cmd = new StringBuilder("-u " + Quote(script));
        foreach (string a in argv) cmd.Append(' ').Append(Quote(a));
        if (!gui) return Exec(python, cmd.ToString(), false);

        string pythonw = Path.Combine(Path.GetDirectoryName(python), "pythonw.exe");
        if (!File.Exists(pythonw)) pythonw = python;
        Process p = Start(pythonw, cmd.ToString(), true);
        // Stay a moment so an early crash is shown instead of silently vanishing.
        if (p.WaitForExit(5000) && p.ExitCode != 0)
        {
            Console.WriteLine("The window failed to start (exit " + p.ExitCode + "). Last log lines:");
            string log = Path.Combine(home, "gui.log");
            if (File.Exists(log))
            {
                string[] lines = File.ReadAllLines(log);
                for (int i = Math.Max(0, lines.Length - 15); i < lines.Length; i++) Console.WriteLine(lines[i]);
            }
            return p.ExitCode;
        }
        noPause = true;
        return 0;
    }

    static bool noPause;

    static Process Start(string exe, string arguments, bool noWindow)
    {
        ProcessStartInfo psi = new ProcessStartInfo(exe, arguments);
        psi.UseShellExecute = false;
        psi.CreateNoWindow = noWindow;
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["HF_HUB_DISABLE_SYMLINKS"] = "1";
        psi.EnvironmentVariables["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1";
        psi.EnvironmentVariables["HF_HUB_DISABLE_PROGRESS_BARS"] = noWindow ? "1" : "0";
        psi.EnvironmentVariables["UV_PYTHON_INSTALL_DIR"] = Path.Combine(home, "python");
        psi.EnvironmentVariables["UV_CACHE_DIR"] = Path.Combine(home, "uv-cache");
        psi.EnvironmentVariables["UV_LINK_MODE"] = "copy";
        return Process.Start(psi);
    }

    static string EnsureEnvironment()
    {
        string venv = Path.Combine(home, "venv");
        string python = Path.Combine(venv, "Scripts", "python.exe");
        string marker = Path.Combine(venv, ".ready-v" + EnvVersion);
        if (File.Exists(python) && File.Exists(marker)) return python;

        Console.WriteLine("First start: setting up (Python + Whisper, about 0.3 GB, or 1.5 GB with an NVIDIA GPU).");
        Console.WriteLine("Install folder: " + home);
        string uv = EnsureUv();

        Exec(uv, "venv --allow-existing --python 3.11 " + Quote(venv), true);
        bool gpu = HasNvidiaGpu();
        Console.WriteLine(gpu ? "NVIDIA GPU found: installing CUDA libraries too." : "No NVIDIA GPU found: will run on the CPU.");
        string pkgs = BasePackages + (gpu ? " " + CudaPackages : "");
        Exec(uv, "pip install --python " + Quote(python) + " " + pkgs, true);
        Exec(uv, "cache clean -q", false); // the env holds copies; the cache would double disk use
        File.WriteAllText(marker, DateTime.Now.ToString("s"));
        Console.WriteLine("Setup done.");
        Console.WriteLine();
        return python;
    }

    static string EnsureUv()
    {
        string dir = Path.Combine(home, "uv");
        string uv = Path.Combine(dir, "uv.exe");
        if (File.Exists(uv)) return uv;
        Directory.CreateDirectory(dir);
        string url = "https://github.com/astral-sh/uv/releases/download/" + UvVersion + "/uv-x86_64-pc-windows-msvc.zip";
        string zip = Path.Combine(dir, "uv.zip");
        Console.WriteLine("Downloading uv " + UvVersion + " ...");
        ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072; // TLS 1.2
        using (WebClient wc = new WebClient()) wc.DownloadFile(url, zip);
        using (ZipArchive z = ZipFile.OpenRead(zip))
        {
            foreach (ZipArchiveEntry e in z.Entries)
            {
                if (e.Name.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
                    e.ExtractToFile(Path.Combine(dir, e.Name), true);
            }
        }
        File.Delete(zip);
        if (!File.Exists(uv)) throw new Exception("uv.exe missing after download");
        return uv;
    }

    static bool HasNvidiaGpu()
    {
        string sys = Environment.GetFolderPath(Environment.SpecialFolder.System);
        if (File.Exists(Path.Combine(sys, "nvidia-smi.exe"))) return true;
        string pf = Environment.GetEnvironmentVariable("ProgramW6432") ?? @"C:\Program Files";
        return File.Exists(Path.Combine(pf, @"NVIDIA Corporation\NVSMI\nvidia-smi.exe"));
    }

    static int Exec(string exe, string arguments, bool mustSucceed)
    {
        using (Process p = Start(exe, arguments, false))
        {
            p.WaitForExit();
            if (mustSucceed && p.ExitCode != 0)
                throw new Exception(Path.GetFileName(exe) + " failed (exit " + p.ExitCode + ")");
            return p.ExitCode;
        }
    }

    static string Quote(string s)
    {
        // Standard Windows command-line quoting (backslashes only escape before a quote).
        if (s.Length > 0 && s.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) return s;
        StringBuilder sb = new StringBuilder("\"");
        int bs = 0;
        foreach (char c in s)
        {
            if (c == '\\') { bs++; continue; }
            if (c == '"') { sb.Append('\\', bs * 2 + 1).Append('"'); bs = 0; continue; }
            sb.Append('\\', bs).Append(c);
            bs = 0;
        }
        sb.Append('\\', bs * 2).Append('"');
        return sb.ToString();
    }

    static List<string> SplitArgs(string line)
    {
        List<string> res = new List<string>();
        StringBuilder cur = new StringBuilder();
        bool inQ = false, any = false;
        foreach (char c in line)
        {
            if (c == '"') { inQ = !inQ; any = true; }
            else if (char.IsWhiteSpace(c) && !inQ) { if (any) { res.Add(cur.ToString()); cur.Length = 0; any = false; } }
            else { cur.Append(c); any = true; }
        }
        if (any) res.Add(cur.ToString());
        return res;
    }
}
