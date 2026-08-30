import os
import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
TARGET_BAT = BASE_DIR / "run.bat"
ICON_ICO = BASE_DIR / "assets" / "app_icon.ico"

ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$DesktopPaths = @(
    [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop),
    (Join-Path $env:USERPROFILE "Desktop"),
    (Join-Path $env:USERPROFILE "OneDrive\\Desktop"),
    (Join-Path $env:USERPROFILE "OneDrive\\바탕 화면")
) | Select-Object -Unique

$TargetFile = "{str(TARGET_BAT)}"
$WorkDir = "{str(BASE_DIR)}"
$IconFile = "{str(ICON_ICO)}"

foreach ($dp in $DesktopPaths) {{
    if (Test-Path $dp) {{
        $ShortcutPath = Join-Path $dp "AI 워터마크 제거기.lnk"
        $sc = $WshShell.CreateShortcut($ShortcutPath)
        $sc.TargetPath = $TargetFile
        $sc.WorkingDirectory = $WorkDir
        $sc.Description = "AI Watermark Remover (Image & Video)"
        if (Test-Path $IconFile) {{
            $sc.IconLocation = $IconFile
        }} else {{
            $sc.IconLocation = "$env:SystemRoot\\System32\\imageres.dll,67"
        }}
        $sc.Save()
        Write-Output "Created: $ShortcutPath"
    }}
}}
"""

ps_file = BASE_DIR / "create_shortcut.ps1"
with open(ps_file, "w", encoding="utf-8-sig") as f:
    f.write(ps_script)

res = subprocess.run(
    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps_file)],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="ignore",
)

if ps_file.exists():
    try:
        ps_file.unlink()
    except Exception:
        pass

print("=== Desktop Shortcut Status ===")
for p in [
    Path(os.environ.get("USERPROFILE", "")) / "Desktop" / "AI 워터마크 제거기.lnk",
    Path(os.environ.get("USERPROFILE", "")) / "OneDrive" / "Desktop" / "AI 워터마크 제거기.lnk",
    Path(os.environ.get("USERPROFILE", "")) / "OneDrive" / "바탕 화면" / "AI 워터마크 제거기.lnk",
]:
    if p.exists():
        print(f"✅ Found shortcut on Desktop: {p}")
