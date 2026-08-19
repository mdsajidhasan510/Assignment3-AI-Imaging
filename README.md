# Assignment 3 — AI Imaging: Auditable Nuclei Analysis

Reproducible repository for fluorescence-microscopy nuclei segmentation using classical image processing, a local Ollama vision model, U-Net, region measurements, and a numbers-first text LLM.

## Pipeline

`image -> grayscale 256x256 -> VLM description`  
`image -> Otsu + morphology -> regionprops`  
`image -> U-Net -> cleanup -> regionprops -> numerical audit -> text LLM -> JSON/CSV`

## Dataset

Do not commit the supplied dataset. Place it at `data/nuclei_dataset/` with `train/images`, `train/masks`, `val/images`, `val/masks`, `test/images`, `test/masks`, plus metadata. Expected split: 80 train, 20 validation, 12 held-out test images.

## Colab / Ollama

The lecturer-provided workaround for the Windows `mllama` error is Google Colab with a T4 GPU. In Colab:

```bash
pip install -r requirements.txt
apt-get update -qq && apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama:

```python
import subprocess, time
p = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
```

Pull the vision model actually used in the final report:

```bash
ollama pull qwen2.5vl:7b
```

If using the lecturer-approved `llama3.2-vision` Colab route instead, pull that model and change the model name in the code. For the text-only stage, use an installed text model, e.g. `llama3.1:8b`.

## Run

```bash
pip install -r requirements.txt
python scripts/run_otsu.py --data-dir data/nuclei_dataset --split val
python scripts/train_unet.py --data-dir data/nuclei_dataset --epochs 12 --batch-size 4 --lr 5e-4 --width 32 --checkpoint models/unet_dice.pt
python scripts/evaluate_unet.py --data-dir data/nuclei_dataset --checkpoint models/unet_dice.pt
```

Run the final test pipeline only after training:

```bash
python scripts/run_test_pipeline.py --data-dir data/nuclei_dataset --checkpoint models/unet_dice.pt --text-model llama3.1:8b
```

## Main reported configuration

3-level U-Net, width 32 (~1.93M parameters), soft Dice loss, Adam, learning rate 5e-4, batch size 4, 12 epochs, flips and right-angle rotations, grayscale 256x256 input.

## Reported results

| Method | Validation Dice | Validation IoU |
|---|---:|---:|
| Otsu baseline | 0.9775 | — |
| Main U-Net, width 32, Dice, 12 epochs | **0.9938** | **0.9877** |
| Width 16, BCE, 6 epochs | 0.9005 | — |
| Width 16, Dice, 6 epochs | 0.8396 | — |
| Width 16, BCE + Dice, 6 epochs | 0.8874 | — |

The supplied report states that U-Net beat Otsu on all 20 validation images. It also reports 12/12 test counts copied correctly by the text model, while density classification was unreliable.

**Important:** these are reported experiment results, not a claim that this repository has regenerated them. Re-run the code before submission and keep generated evidence.

## Repository structure

```text
README.md
requirements.txt
.gitignore
src/
  data.py
  classical.py
  metrics.py
  unet.py
  prompts.py
  ollama_utils.py
  pipeline.py
scripts/
  run_otsu.py
  train_unet.py
  evaluate_unet.py
  run_test_pipeline.py
notebooks/
  Assignment_3_AI_Imaging.ipynb
outputs/
models/
docs/RESULTS.md
```

## Trust and limitations

The segmentation measurements are treated as the source of truth. The LLM is not allowed to invent quantitative values. The final CSV keeps measured audit columns beside model fields so hallucinated categorical judgements can be detected. The dataset is synthetic and clean; the report correctly concludes that clinical use is not justified without evaluation on real annotated microscopy.
