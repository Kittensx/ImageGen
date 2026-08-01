import os
import subprocess
import sys
from pathlib import Path

# IMAGE_GEN_ORGANIZED_PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements" / "requirements.txt"

PATCHED_NUMPY_PATH = os.path.join(PROJECT_ROOT, "imagecore", "repositories", "numpy")

def get_venv_python():
    if os.name == "nt":
        return str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    else:
        return str(PROJECT_ROOT / "venv" / "bin" / "python")

def install_requirements_without_numpy(python_exec):
    if not REQUIREMENTS_FILE.exists():
        print(f"❌ Requirements file not found: {REQUIREMENTS_FILE}")
        return

    print("🔄 Installing packages from requirements.txt (excluding numpy)...")
    with REQUIREMENTS_FILE.open("r", encoding="utf-8") as f:
        lines = [line for line in f if not line.lower().startswith("numpy")]

    temp_file = str(PROJECT_ROOT / "requirements" / "temp_requirements.txt")
    with open(temp_file, "w") as f:
        f.writelines(lines)

    subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([python_exec, "-m", "pip", "install", "-r", temp_file], check=True)
    os.remove(temp_file)
    print("✅ Updated venv (excluding numpy).")

def install_patched_numpy(python_exec):
    if not os.path.exists(PATCHED_NUMPY_PATH):
        print(f"⚠️ Patched numpy not found at {PATCHED_NUMPY_PATH}")
        return
    print(f"Installing local patched numpy from {PATCHED_NUMPY_PATH}...")
    subprocess.run([python_exec, "-m", "pip", "install", "-e", PATCHED_NUMPY_PATH], check=True)
    print("✅ Patched numpy installed.")

def main():
    python_exec = get_venv_python()
    if not os.path.exists(python_exec):
        print("❌ Virtual environment not found. Run install_venv.py first.")
        return
    install_requirements_without_numpy(python_exec)
    install_patched_numpy(python_exec)

if __name__ == "__main__":
    main()
