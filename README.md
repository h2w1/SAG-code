
# Selective Critique is Enough: Cost-Aware Feedback for LLM Agents in Long-Horizon Interactive Environments

This repository contains the **reproduction notebooks** and minimal environment setup for three benchmarks used in our paper:

- **ALFWorld** – household instruction following in simulated environments [site](https://alfworld.github.io/)
- **WebShop** – goal‑oriented web browsing / shopping [site](https://webshop-pnlp.github.io/)
- **BabyAI** – grid‑world language‑conditioned tasks [site](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)

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

---

## 1) ALFWorld

### Create & activate environment
```bash
conda create -n alfworld python=3.9 -y
conda activate alfworld
```
### Install ALFWorld from source
Follow the official instructions and prefer the source installation:
- Repo: https://github.com/alfworld/alfworld


### Install Python dependencies
```bash
pip install -r envs/alfworld/requirements.txt
```
---


## 2) WebShop

### Create & activate environment
```bash
conda create -n webshop python=3.9 -y
conda activate webshop
```
### Install WebShop from source
Follow the official instructions (install from source is recommended):
- Repo: https://github.com/princeton-nlp/WebShop

### Install Python dependencies
```bash
pip install -r envs/webshop/requirements.txt
```


---

## 3) BabyAI

### Install BabyAI environment

This codebase relies on BabyAI and its text-based extensions.
Please install them separately before installing other dependencies.

Example:

### Install WebShop from source
Follow the official instructions (install from source is recommended):
- Repo: https://github.com/flowersteam/Grounding_LLMs_with_online_RL

### Install Python dependencies
```bash
pip install -r envs/babyai/requirements.txt
```



---

## 4) Running the python code



---
