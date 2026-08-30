import os
import sys
import urllib.request
from pathlib import Path

WEIGHTS = {
    "ProPainter.pth": [
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        "https://huggingface.co/sczhou/ProPainter/resolve/main/ProPainter.pth",
    ],
    "recurrent_flow_completion.pth": [
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "https://huggingface.co/sczhou/ProPainter/resolve/main/recurrent_flow_completion.pth",
    ],
    "raft-things.pth": [
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        "https://huggingface.co/sczhou/ProPainter/resolve/main/raft-things.pth",
    ],
    "big-lama.pt": [
        "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "https://huggingface.co/smartcomputer/big-lama/resolve/main/big-lama.pt",
        "https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.pt",
    ],
}


def download_with_progress(url: str, save_path: Path):
    print(f"Downloading {save_path.name} from {url}...")
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            },
        )
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1MB
            downloaded = 0

            temp_path = save_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(
                            f"\r[{percent:5.1f}%] {mb_down:.1f} MB / {mb_total:.1f} MB"
                        )
                        sys.stdout.flush()
            if total_size > 0:
                sys.stdout.write("\n")
            if temp_path.exists():
                if save_path.exists():
                    save_path.unlink()
                temp_path.rename(save_path)
            print(f"Successfully downloaded {save_path.name} ({downloaded / (1024 * 1024):.1f} MB)")
            return True
    except Exception as e:
        print(f"\nFailed to download from {url}: {e}")
        temp_path = save_path.with_suffix(".tmp")
        if temp_path.exists():
            temp_path.unlink()
        return False


def ensure_single_weight(filename: str, target_dirs=None) -> bool:
    if target_dirs is None:
        base_dir = Path(__file__).resolve().parent
        target_dirs = [
            base_dir / "weights",
            base_dir / "ProPainter" / "weights",
        ]

    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

    urls = WEIGHTS.get(filename, [])
    if not urls:
        print(f"[ERROR] Unknown weight filename: {filename}")
        return False

    primary_file = target_dirs[0] / filename
    if primary_file.exists() and primary_file.stat().st_size > 1024 * 1024:
        print(f"[OK] {filename} already exists ({primary_file.stat().st_size / (1024 * 1024):.1f} MB)")
        success = True
    else:
        success = False
        for url in urls:
            if download_with_progress(url, primary_file):
                success = True
                break
        if not success:
            print(f"[ERROR] Could not download {filename}")

    # Mirror to other target directories
    for d in target_dirs[1:]:
        dest_file = d / filename
        if not dest_file.exists() and primary_file.exists():
            try:
                import shutil
                shutil.copyfile(primary_file, dest_file)
            except Exception as e:
                print(f"Warning: could not mirror {filename} to {dest_file}: {e}")

    return success


def ensure_lama_weights(target_dirs=None) -> bool:
    return ensure_single_weight("big-lama.pt", target_dirs=target_dirs)


def ensure_propainter_weights(target_dirs=None) -> bool:
    all_ok = True
    for fn in ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]:
        if not ensure_single_weight(fn, target_dirs=target_dirs):
            all_ok = False
    return all_ok


def ensure_weights(target_dirs=None):
    if target_dirs is None:
        base_dir = Path(__file__).resolve().parent
        target_dirs = [
            base_dir / "weights",
            base_dir / "ProPainter" / "weights",
        ]

    all_success = True
    for filename in WEIGHTS.keys():
        if not ensure_single_weight(filename, target_dirs=target_dirs):
            all_success = False

    return all_success


if __name__ == "__main__":
    success = ensure_weights()
    if success:
        print("\nAll weights (ProPainter + LaMa) are ready!")
    else:
        print("\nFailed to download some weights.")
        sys.exit(1)
