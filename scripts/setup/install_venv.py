import os
import sys
import subprocess
import venv
from pathlib import Path

# IMAGE_GEN_ORGANIZED_PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements" / "requirements.txt"

VENV_DIR = str(PROJECT_ROOT / "venv")


def create_virtual_environment():
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in '{VENV_DIR}'...")
        venv.create(VENV_DIR, with_pip=True)
        print("✅ Virtual environment created.")
    else:
        print(f"Virtual environment '{VENV_DIR}' already exists.")

def get_venv_python():
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")



def install_requirements(python_exec):
    print("🔧 Installing dependencies from requirements.txt...")

    if not REQUIREMENTS_FILE.exists():
        print(f"❌ Requirements file not found: {REQUIREMENTS_FILE}")
        return

    try:
        # Upgrade pip
        subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=True)

        # Install from requirements.txt
        subprocess.run([python_exec, "-m", "pip", "install", "--no-cache-dir", "-r", str(REQUIREMENTS_FILE)], check=True)

        print("✅ All dependencies installed successfully.")

    except subprocess.CalledProcessError as e:
        print("❌ An error occurred during installation.")
        print("Details:", e)



def main():
    create_virtual_environment()
    python_exec = get_venv_python()
    install_requirements(python_exec)
    

if __name__ == "__main__":
    main()
