# SAG: Self-improving Agent with Gated critique

Code for **Selective Critique for Cost-Aware LLM Agents in Long-Horizon Decision Making** (NeurIPS 2026).

[Project page](https://h2w1.github.io/SAG/)

SAG selectively invokes critique using action-level ambiguity signals and improves the actor online from successful trajectories. This repository contains Python implementations for ALFWorld, WebShop, and BabyAI.

## Files

```text
SAG_alfworld.py
SAG_webshop.py
SAG_babyai.py
prompots/alfworld_3prompts_1.json
envs/alfworld/requirements.txt
envs/webshop/requirements.txt
envs/webshop/jobs/sag.sh
envs/babyai/requirements.txt
```

The prompt directory is currently named `prompots`. The entry points are the three `.py` scripts.

## Setup

Use separate Conda environments for the three benchmarks. The examples below use Python 3.10. A CUDA GPU with sufficient memory for the selected model is recommended. Install a PyTorch build appropriate for your CUDA version.

```bash
git clone https://github.com/h2w1/SAG-code.git
cd SAG-code
```

Run all SAG commands from this repository root. ALFWorld and BabyAI set `CUDA_VISIBLE_DEVICES` inside their scripts (currently `0` and `2`, respectively); adjust those values for your machine.

### ALFWorld

Install [ALFWorld](https://github.com/alfworld/alfworld) and download its game data according to the upstream instructions. Then install the agent dependencies:

```bash
conda create -n sag-alfworld python=3.10 -y
conda activate sag-alfworld
# Install ALFWorld in this environment following its upstream instructions.
pip install -r envs/alfworld/requirements.txt
pip install torch transformers accelerate pyyaml
```

The script reads `base_config.yaml` from the current directory. Copy `configs/base_config.yaml` from your ALFWorld source checkout to this repository root and configure its data paths and text environment (`AlfredTWEnv`). Prepare the supplied prompts at the path expected by the script:

```bash
mkdir -p prompts
cp prompots/alfworld_3prompts_1.json prompts/alfworld_prompts.json
python SAG_alfworld.py
```

The current defaults use `Qwen/Qwen2.5-7B-Instruct` and 134 episodes on `eval_out_of_distribution`. Change `MODEL_NAME` and the evaluation loop in the script to select another configuration. Results, trajectories, and SFT samples are written to `trajectories/`.

### WebShop

Set up the [WebShop environment](https://github.com/princeton-nlp/WebShop), including its product data and search index. Start its web server using the upstream `run_dev.sh` instructions, normally on port 3000. The environment server may use its own environment; install SAG agent dependencies separately:

```bash
conda create -n sag-webshop python=3.10 -y
conda activate sag-webshop
pip install -r envs/webshop/requirements.txt
pip install torch transformers accelerate peft requests beautifulsoup4
python SAG_webshop.py --port 3000 --model_name_or_path Qwen/Qwen2.5-7B-Instruct
```

Set `--port` to the running WebShop server port. The current entry point runs 500 episodes with seed 42; change `n` in the final `run_episodes` call for another episode count. Results are saved to `result/`, with SFT data in `trajectories/`.

### BabyAI

Install BabyAI and BabyAI-Text following the [upstream installation instructions](https://github.com/flowersteam/Grounding_LLMs_with_online_RL), including the `babyai-text` package. The script imports `babyai_text` and uses `gym.make("BabyAI-GoToLocal-v0")`.

```bash
conda create -n sag-babyai python=3.10 -y
conda activate sag-babyai
# Install BabyAI and BabyAI-Text in this environment first.
pip install -r envs/babyai/requirements.txt
python SAG_babyai.py
```

The current defaults use `Qwen/Qwen2.5-7B-Instruct`, the `goto` task, 50 episodes, and seed 42. Set `model_name_or_path`, `task_type`, `num_task`, and `start_idx` in the main block for the desired configuration. Summary results and detailed logs are saved as `react_final_<model>_<task>.json` and `react_log_<model>_<task>.json`.

## Experiment configuration

Gate thresholds, success-demonstration settings, and online LoRA/SFT settings are defined in each benchmark script. Select the model, episode count, task split, and these settings for the experiment you want to run. The default commands above describe the checked-in configuration; they do not select every model or ablation reported in the paper.

The environment requirements are benchmark-specific and are not a complete lockfile. After a successful setup, record exact installed versions with `pip freeze` for reproducibility.
