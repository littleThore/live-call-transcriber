@echo off
rem Builds dist\LiveCallTranscriber.exe with the C# compiler that ships with Windows (.NET Framework 4.x).
setlocal
cd /d "%~dp0"
if not exist dist mkdir dist
"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /optimize+ /target:exe ^
  /out:dist\LiveCallTranscriber.exe ^
  /resource:live_transcribe.py,live_transcribe.py /resource:gui.py,gui.py ^
  /r:System.IO.Compression.dll /r:System.IO.Compression.FileSystem.dll ^
  launcher\Launcher.cs
if errorlevel 1 exit /b 1
echo Built dist\LiveCallTranscriber.exe
