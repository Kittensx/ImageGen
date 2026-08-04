"""Rebuild IMAGE_GEN through the validated hardware-aware installer.

The former updater modified a legacy environment without checking Torch, CUDA,
MSLK, or xFormers compatibility. Updates now use the same scan, rollback, and
validation path as a clean installation.
"""

from install_image_gen import main


if __name__ == "__main__":
    raise SystemExit(main())
