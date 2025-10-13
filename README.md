
# Confidence-gated Intervention for LLM Reasoning in Multi-Step Environments with Player–Coach Agents

This repository contains the **reproduction notebooks** and minimal environment setup for three benchmarks used in our paper:

- **ALFWorld** – household instruction following in simulated environments
- **WebShop** – goal‑oriented web browsing / shopping
- **BabyAI** – grid‑world language‑conditioned tasks

Each benchmark is isolated in its own Conda environment to avoid dependency conflicts. We keep the `requirements.txt` minimal and install the official benchmark packages **from source** following their instructions.

> If you find version mismatches on your machine, please pin exact versions by running `pip freeze > envs/<benchmark>/requirements.lock.txt` after a successful setup on your system.

---

## File Structure

```
.
├─ ALFWorld.ipynb          # Reproduction notebook for ALFWorld
├─ prompts/alfworld_3prompts_1.json #prompts for ALFWorld 
├─ Webshop.ipynb           # Reproduction notebook for WebShop
├─ BabyAI.ipynb            # Reproduction notebook for BabyAI
├─ envs/
│  ├─ alfworld/requirements.txt   # Minimal Python deps for ALFWorld env (benchmark installed from source)
│  ├─ webshop/requirements.txt    # Minimal Python deps for WebShop env (benchmark installed from source)
│  └─ babyai/requirements.txt     # Minimal Python deps for BabyAI env (benchmark installed from source)
└─ README.md               # This file
```

> The three `.ipynb` files are the main entry points. Open them after completing the environment setup below.

---

## 0) Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Python 3.9** (we use this version in all commands below)
- A C/C++ build toolchain may be required by some upstream repos

---

## 1) ALFWorld

### Create & activate environment
```bash
conda create -n alfworld python=3.9 -y
conda activate alfworld
```

### Install Python dependencies
```bash
pip install -r envs/alfworld/requirements.txt
```

### Install ALFWorld from source
Follow the official instructions and prefer the source installation:
- Repo: https://github.com/alfworld/alfworld

> Tip: Some ALFWorld setups require additional assets or simulators. Please follow the upstream README carefully.

---

## 2) WebShop

### Create & activate environment
```bash
conda create -n webshop python=3.9 -y
conda activate webshop
```

### Install Python dependencies
```bash
pip install -r envs/webshop/requirements.txt
```

### Install WebShop from source
Follow the official instructions (install from source is recommended):
- Repo: https://github.com/princeton-nlp/WebShop

> Tip: If the environment name was previously `babyai` in your notes, you can keep using `webshop` here to avoid confusion.

---

## 3) BabyAI

### Create & activate environment
```bash
conda create -n babyai python=3.9 -y
conda activate babyai
```

### Install Python dependencies
```bash
pip install -r envs/babyai/requirements.txt
```

### Install BabyAI from source
Follow the official instructions to install the benchmark:
- Repo: https://github.com/mila-iqia/babyai

> Tip: BabyAI may require specific `gym` versions depending on the branch. Use the upstream README’s version matrix if you encounter issues.

---

## 4) Running the Notebooks

After finishing the setup for a benchmark, start Jupyter and select the corresponding Conda kernel.

```bash
# Example for ALFWorld
conda activate alfworld
python -m ipykernel install --user --name alfworld --display-name "Python (alfworld)"
jupyter notebook  # or: jupyter lab
```

Then open the notebook file (`ALFWorld.ipynb`, `Webshop.ipynb`, or `BabyAI.ipynb`) and run all cells.

---
