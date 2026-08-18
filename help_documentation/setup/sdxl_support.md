# Stable Diffusion XL Runtime Support

IMAGE_GEN uses lightweight Hugging Face runtime assets to describe SDXL architecture, tokenizer, scheduler, and component configuration. The user's selected checkpoint continues to provide the heavy diffusion-model weights.

## Installer

From the IMAGE_GEN project root on Windows, run:

```bat
install_sdxl_support.bat
```

The default install acquires support assets for:

- SDXL Base 1.0 -> `runtime_assets/stable_diffusion/SDXL_Base`
- SDXL Base Refiner 1.0 -> `runtime_assets/stable_diffusion/SDXL_Base_Refiner`
- SDXL Turbo -> `runtime_assets/stable_diffusion/SDXL_Turbo`

The installer uses official Stability AI repositories on Hugging Face and downloads only lightweight architecture/config/tokenizer files. It deliberately rejects checkpoint, UNet, VAE, ONNX, and text-encoder weight files.

## Install one support tree

```bat
install_sdxl_support.bat --profile base
install_sdxl_support.bat --profile refiner
install_sdxl_support.bat --profile turbo
```

## Japanese SDXL support assets

The project also recognizes the existing runtime-asset folder names:

- `runtime_assets/stable_diffusion/japanese-stable-diffusion-xl`
- `runtime_assets/stable_diffusion/japanese-stable-clip-vit-l-16`

Those Stability AI Hugging Face repositories are gated. First accept the repository access terms while signed into Hugging Face, then authenticate locally:

```bat
.venv\Scripts\hf.exe auth login
```

If the `hf` command is not exposed through the virtual environment, use the normal Hugging Face login mechanism available in the installed environment or set the `HF_TOKEN` environment variable.

Then run:

```bat
install_sdxl_support.bat --profile japanese
```

Or install the normal Base/Refiner/Turbo assets and the Japanese support files together:

```bat
install_sdxl_support.bat --include-japanese
```

The Japanese option installs lightweight config, tokenizer, and official custom-code support files only. It does not download the multi-gigabyte model or text-encoder weights.

## Check or preview installation

Check whether the selected files already exist:

```bat
install_sdxl_support.bat --status-only
```

Preview download destinations without changing files:

```bat
install_sdxl_support.bat --dry-run
```

Force the selected support files to be downloaded again and atomically replaced:

```bat
install_sdxl_support.bat --refresh
```

Without `--refresh`, existing non-empty files are preserved and only missing files are downloaded.

## Installed support receipt

After a successful install, IMAGE_GEN writes:

```text
artifacts/install/sdxl_support_receipt.json
```

The receipt records the selected support sets, resolved runtime-assets root, downloaded files, and lightweight architecture validation results.

## Weight ownership

The SDXL support installer is intentionally not a model installer. It will not download `.safetensors`, `.ckpt`, `.bin`, `.pt`, `.pth`, `.onnx`, or `.gguf` weight files into `runtime_assets`.

Keep normal SDXL checkpoints in the project's configured model library. Runtime support assets and user model weights remain separate responsibilities.
