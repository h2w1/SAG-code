import os
import sys
import json
import yaml
import random
import re
import logging
import warnings
from pathlib import Path
from collections import deque
from difflib import get_close_matches
from typing import List, Optional, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# =========================================================
# GPU / Logging
# =========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# =========================================================
# Save buffers
# =========================================================
SAVE_DIR = Path("./trajectories")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

success_steps_list: List[int] = []
success_trajectories: List[dict] = []
failure_trajectories: List[dict] = []
decision_logs: List[dict] = []


def save_all_json() -> None:
    (SAVE_DIR / "success_steps.json").write_text(
        json.dumps(success_steps_list, indent=2, ensure_ascii=False)
    )
    (SAVE_DIR / "success.json").write_text(
        json.dumps(success_trajectories, indent=2, ensure_ascii=False)
    )
    (SAVE_DIR / "failure.json").write_text(
        json.dumps(failure_trajectories, indent=2, ensure_ascii=False)
    )
    (SAVE_DIR / "decision_analysis.json").write_text(
        json.dumps(decision_logs, indent=2, ensure_ascii=False)
    )


# =========================================================
# Token counters
# =========================================================
TOTAL_PROMPT_TOKENS = 0
TOTAL_GEN_TOKENS = 0


def _add_tokens(prompt_tokens: int, gen_tokens: int) -> None:
    global TOTAL_PROMPT_TOKENS, TOTAL_GEN_TOKENS
    TOTAL_PROMPT_TOKENS += int(prompt_tokens)
    TOTAL_GEN_TOKENS += int(gen_tokens)


# =========================================================
# Load HF LLM (Actor / Critic backbone)
# =========================================================
# Possible choices:
#   "meta-llama/Llama-3.1-8B-Instruct"
#   "mistralai/Mistral-7B-Instruct-v0.3"
#   "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading LLM ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def llm(prompt: str, stop: List[str] = ["\n"], max_new_tokens: int = 100) -> str:
    """Deterministic decoding wrapper (used for both actor and critic)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    gen_tokens = int(gen_ids.shape[0])
    _add_tokens(prompt_tokens, gen_tokens)

    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

    if stop is not None:
        for s in stop:
            idx = generated.find(s)
            if idx != -1:
                generated = generated[:idx]
                break

    return generated.strip()


# =========================================================
# ALFWorld env
# =========================================================
import alfworld
from alfworld.agents.environment import get_environment

with open("base_config.yaml") as f:
    config = yaml.safe_load(f)

split = "eval_out_of_distribution"
env_name = config["env"]["type"]
EnvClass = get_environment(env_name)
env = EnvClass(config, train_eval=split)
env = env.init_env(batch_size=1)


def process_ob(ob: str) -> str:
    """Normalize ALFWorld observations."""
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob.strip()


def unpack_info(info):
    return info[0] if isinstance(info, list) else info


# =========================================================
# Valid actions flattening
# =========================================================
def to_string_actions(x) -> List[str]:
    """Flatten nested admissible action structures into a unique string list."""
    out: List[str] = []

    def rec(obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple, set)):
            for z in obj:
                rec(z)
        elif isinstance(obj, str):
            out.append(obj)
        else:
            try:
                out.append(str(obj))
            except Exception:
                pass

    rec(x)

    seen = set()
    flat: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            flat.append(s)
    return flat


def get_valid_actions(info) -> List[str]:
    info = unpack_info(info)
    raw = info.get("admissible_commands", [])
    return to_string_actions(raw)


# =========================================================
# Canonicalization utilities
# =========================================================
SURFACE_NAMES = {
    "sofa",
    "coffeetable",
    "sidetable",
    "table",
    "countertop",
    "shelf",
    "armchair",
    "chair",
    "bed",
    "dresser",
    "stoveburner",
    "toaster",
    "sinkbasin",
}
CONTAINER_NAMES = {
    "cabinet",
    "drawer",
    "fridge",
    "microwave",
    "safe",
    "garbagecan",
    "dishwasher",
}


def noun_id(s: str) -> Tuple[str, str]:
    """Extract (noun, id) from ALFWorld object string, if present."""
    m = re.search(r"([a-zA-Z_]+)\s+(\d+)", s.strip().lower())
    return (m.group(1), m.group(2)) if m else (s.strip().lower(), "")


def canonicalize_action(candidate: str, valid_actions: List[str]) -> Optional[str]:
    """Map natural-language-like action into an admissible ALFWorld action."""
    valid_actions = to_string_actions(valid_actions)
    cand = candidate.strip().lower()
    valid_low = [v.lower() for v in valid_actions]

    # 1) Exact match
    if cand in valid_low:
        return valid_actions[valid_low.index(cand)]

    # 2) MOVE pattern
    m_mv = re.search(r"(?:move)\s+(.+?)\s+(?:to)\s+(.+)$", cand)
    if m_mv:
        obj_txt, tgt_txt = m_mv.group(1).strip(), m_mv.group(2).strip()
        t_noun, _ = noun_id(tgt_txt)
        move_cands = [v for v in valid_actions if v.lower().startswith("move ")]
        for v in move_cands:
            if obj_txt in v.lower() and t_noun in v.lower():
                return v
        for v in move_cands:
            if t_noun in v.lower():
                return v
        if move_cands:
            return move_cands[0]

    # 3) PUT / PLACE / MOVE → normalize to put {obj} on|in {tgt}
    m = re.search(r"(put|place|move)\s+(.+?)\s+(?:in|on|into|onto|to)\s+(.+)$", cand)
    if m:
        obj, tgt = m.group(2).strip(), m.group(3).strip()
        t_noun, _ = noun_id(tgt)
        prep = "on" if t_noun in SURFACE_NAMES else ("in" if t_noun in CONTAINER_NAMES else "on")
        norm = f"put {obj} {prep} {tgt}"
        if norm in valid_low:
            return valid_actions[valid_low.index(norm)]
        cands = [
            v
            for v in valid_actions
            if v.lower().startswith("put ") and obj in v.lower() and t_noun in v.lower()
        ]
        if cands:
            return cands[0]

    # 4) drop / leave
    dm = re.search(r"(drop|leave)\s+(.+)$", cand)
    if dm:
        obj = dm.group(2).strip()
        for v in valid_actions:
            if v.lower().startswith("drop ") and obj in v.lower():
                return v

    # 5) go to
    gm = re.search(r"(go to|goto)\s+(.+)$", cand)
    if gm:
        tgt = gm.group(2).strip().lower()
        gotos = [v for v in valid_actions if v.lower().startswith("go to ")]
        close = get_close_matches(f"go to {tgt}", [g.lower() for g in gotos], n=1)
        if close:
            return gotos[[g.lower() for g in gotos].index(close[0])]

    # 6) general fuzzy match
    close = get_close_matches(cand, valid_low, n=1)
    if close:
        return valid_actions[valid_low.index(close[0])]
    return None


def ensure_at_target(env, info, target: str):
    """Ensure the agent has navigated to the target object (if possible)."""
    valid = get_valid_actions(info)
    low = [v.lower() for v in valid]
    want = f"go to {target}".lower()
    if want in low:
        act = valid[low.index(want)]
        return env.step([act])
    return None, None, None, info


def verify_on_target(env, info, obj: str, target: str):
    """Verify that the object is placed correctly on / in the target."""
    valid = get_valid_actions(info)
    low = [v.lower() for v in valid]

    exa = f"examine {target}".lower()
    if exa in low:
        act = valid[low.index(exa)]
        obs2, sc2, dn2, info2 = env.step([act])
        ok = obj.split()[0].lower() in obs2[0].lower()
        return obs2, sc2, dn2, info2, ok

    inv = [v for v in valid if v.lower().startswith("inventory")]
    if inv:
        obs2, sc2, dn2, info2 = env.step([inv[0]])
        ok = obj.split()[0].lower() not in obs2[0].lower()
        return obs2, sc2, dn2, info2, ok

    return None, None, None, info, None


def put_with_fallback(env, info, obj: str, target: str):
    """
    Execute put/move/drop with a robust fallback strategy and verification.
    This is an environment-specific helper and does not change the gating logic.
    """
    steps_used = 0
    subtrace: List[dict] = []

    res = ensure_at_target(env, info, target)
    if res[0] is not None:
        obs_go, sc_go, dn_go, info = res
        steps_used += 1
        subtrace.append(
            {
                "type": "act",
                "exec": f"go to {target}",
                "obs": process_ob(obs_go[0]),
                "reward": sc_go[0] if isinstance(sc_go, list) else sc_go,
                "done": dn_go[0] if isinstance(dn_go, list) else dn_go,
            }
        )
    else:
        info = res[3]

    valid = get_valid_actions(info)

    def _run_and_check(action_str):
        nonlocal steps_used, subtrace
        obs, sc, dn, inf = env.step([action_str])
        steps_used += 1
        subtrace.append(
            {
                "type": "act",
                "exec": action_str,
                "obs": process_ob(obs[0]),
                "reward": sc[0] if isinstance(sc, list) else sc,
                "done": dn[0] if isinstance(dn, list) else dn,
            }
        )
        nothing_happens = "nothing happens" in obs[0].lower()
        obs2, sc2, dn2, inf2, ok = verify_on_target(env, inf, obj, target)
        if obs2 is not None:
            steps_used += 1
            subtrace.append(
                {
                    "type": "act",
                    "exec": "verify_on_target",
                    "obs": process_ob(obs2[0]),
                    "reward": sc2[0] if isinstance(sc2, list) else sc2,
                    "done": dn2[0] if isinstance(dn2, list) else dn2,
                }
            )
        return (obs2 or obs), (sc2 or sc), (dn2 or dn), (inf2 or inf), (ok is True), nothing_happens

    move_cand = canonicalize_action(f"move {obj} to {target}", valid)
    if move_cand:
        obs, sc, dn, inf, ok_m, nh_m = _run_and_check(move_cand)
        if ok_m and not nh_m:
            return obs, sc, dn, inf, True, steps_used, subtrace

    t_noun, _ = noun_id(target)
    pref = "on" if t_noun in SURFACE_NAMES else ("in" if t_noun in CONTAINER_NAMES else "on")
    put_cand = (
        canonicalize_action(f"put {obj} {pref} {target}", valid)
        or canonicalize_action(f"put {obj} on {target}", valid)
        or canonicalize_action(f"put {obj} in {target}", valid)
    )
    if put_cand:
        obs, sc, dn, inf, ok_p, nh_p = _run_and_check(put_cand)
        if ok_p and not nh_p:
            return obs, sc, dn, inf, True, steps_used, subtrace

    drop_cand = canonicalize_action(f"drop {obj}", valid)
    if drop_cand:
        obs, sc, dn, inf, ok_d, nh_d = _run_and_check(drop_cand)
        return obs, sc, dn, inf, bool(ok_d), steps_used, subtrace

    raise RuntimeError(f"No admissible action to place ({obj} -> {target}).")


# =========================================================
# Ambiguity-based gating (critic invocation)
# =========================================================
GATE_CFG = {
    "enable": True,
    "min_actions": 2,
    "cooldown": 4,  # minimum steps between critic calls

    # VoI-inspired parameters: entropy term, margin bonus, and critic cost
    "entropy_weight": 1.0,
    "critic_cost": 0.6,
    "margin_threshold": 0.4,
    "margin_bonus": 0.5,
}


@torch.no_grad()
def compute_ambiguity_signals(prefix: str, admissible: List[str]) -> Tuple[float, float, float]:
    """
    Compute ambiguity signals from the actor distribution:
    - H: (unnormalized) entropy over admissible actions (global ambiguity)
    - log_margin: log-prob margin between top-1 and top-2 actions (local ambiguity)
    - pmax: max probability
    """
    if len(admissible) < GATE_CFG["min_actions"]:
        return 0.0, 999.0, 0.0

    log_likelihoods: List[float] = []
    enc_pref = tokenizer(prefix, return_tensors="pt").to(device)
    L_pref = enc_pref["input_ids"].shape[1]

    for a in admissible:
        full_txt = prefix + a
        enc_full = tokenizer(full_txt, return_tensors="pt").to(device)
        out = model(**enc_full)
        logits = out.logits.float()

        logp = 0.0
        L_full = enc_full["input_ids"].shape[1]
        for t in range(L_pref, L_full):
            token_id = enc_full["input_ids"][0, t]
            logp += torch.log_softmax(logits[0, t - 1, :], dim=-1)[token_id].item()
        log_likelihoods.append(logp)

    ll_t = torch.tensor(log_likelihoods)
    prob = torch.softmax(ll_t, dim=0)
    entropy = -(prob * torch.log(prob + 1e-8)).sum().item()

    if ll_t.numel() > 1:
        top2 = torch.topk(ll_t, 2).values
        log_margin = (top2[0] - top2[1]).item()
    else:
        log_margin = 999.0

    return float(entropy), float(log_margin), float(prob.max().item())


def should_invoke_critic(
    prefix: str,
    admissible: List[str],
    step_idx: int,
    episode_name: str,
    to_print: bool = False,
) -> bool:
    """
    Decide whether to invoke the critic using a VoI-inspired ambiguity gate:
        U = entropy_weight * H + margin_bonus * I[log_margin < threshold] - critic_cost
    """
    if not GATE_CFG["enable"]:
        return False
    if len(admissible) < GATE_CFG["min_actions"]:
        return False

    H, log_margin, pmax = compute_ambiguity_signals(prefix, admissible)
    expected_gain = GATE_CFG["entropy_weight"] * H

    margin_bonus = 0.0
    if log_margin < GATE_CFG["margin_threshold"]:
        margin_bonus = GATE_CFG["margin_bonus"]

    cost = GATE_CFG["critic_cost"]
    total_utility = expected_gain + margin_bonus - cost
    trigger = total_utility > 0

    decision_logs.append(
        {
            "episode": episode_name,
            "step": step_idx,
            "entropy": round(H, 4),
            "log_margin": round(log_margin, 4),
            "pmax": round(pmax, 4),
            "expected_gain": round(expected_gain, 4),
            "margin_bonus": margin_bonus,
            "cost": cost,
            "total_utility": round(total_utility, 4),
            "triggered": trigger,
        }
    )

    if to_print:
        mark = " <--- [CRITIC INVOKED]" if trigger else ""
        print(
            f"   >>> [GATE] U={total_utility:.3f} | Gain={expected_gain:.3f} | Cost={cost}{mark}"
        )

    return trigger


# =========================================================
# Critic (ambiguity-gated)
# =========================================================
def _extract_json_block(txt: str) -> dict:
    """Extract the last valid JSON object from a text span."""
    candidates = []
    for m in re.finditer(r"\{.*?\}", txt, flags=re.DOTALL):
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            candidates.append(obj)
        except Exception:
            continue
    return candidates[-1] if candidates else {}


CRITIC_STEP_PROMPT = """You are a critic for an LLM-based agent.
Your GOAL: detect and correct hallucinations, loops, or inconsistent actions.

INPUTS:
- TASK: The goal.
- HISTORY: What happened so far.
- ACTIONS: Valid moves.

INSTRUCTIONS:
1. VERIFY OBJECTS: Does the object the agent is targeting MATCH the task description?
2. VERIFY HISTORY: Did the agent already check this empty location or repeat failures?
3. DECISION:
   - If the agent is SAFE and CORRECT, output "next_action": "continue".
   - If the agent is WRONG, provide a specific "next_action" to FIX it.

IMPORTANT FORMAT RULES:
- Return JSON only.
- "next_action" MUST be a valid action string or "continue".
"""


def _make_trajectory_snippet(steps: List[dict], limit_chars: int = 1800) -> str:
    """Compact textual summary of recent trajectory steps for the critic."""
    parts: List[str] = []
    for st in reversed(steps[-40:]):
        if st["type"] == "think":
            parts.append(f"think: {st.get('llm','')}")
        else:
            parts.append(f"Act: {st.get('exec','')}\nObs: {st.get('obs','')}")
    s = "\n".join(parts[::-1])
    return s[-limit_chars:]


def critic_step_advice(task_line: str, traj_snippet: str, admissible: List[str]) -> dict:
    """Single-step critic query conditioned on trajectory prefix and admissible actions."""
    admissible_block = "\n".join(f"- {a}" for a in admissible)
    prompt = (
        f"{CRITIC_STEP_PROMPT}\n\n"
        f"TASK:\n{task_line}\n\n"
        f"HISTORY:\n{traj_snippet}\n\n"
        f"ACTIONS:\n{admissible_block}\n\n"
        f"JSON:"
    )
    text = llm(prompt, stop=[], max_new_tokens=400)
    data = _extract_json_block(text)

    na = data.get("next_action")
    if isinstance(na, str):
        na = na.strip()
        if na.lower() == "continue":
            data["next_action"] = "continue"
        elif na in admissible:
            data["next_action"] = na
        else:
            cleaned = re.sub(r"^[Aa]ction\s*:\s*", "", na).strip()
            fixed = canonicalize_action(cleaned, admissible)
            data["next_action"] = fixed if fixed else "continue"
    else:
        data["next_action"] = "continue"

    return data


# =========================================================
# Demo buffer for successful trajectories (in-context examples)
# =========================================================
DEMO_CFG = {
    "enable": True,
    "per_task_max": 8,
    "inject_k": 2,          # at most k recent demos per task family
    "max_chars_each": 1800, # limit per formatted demo
}

SUCCESS_DEMOS: Dict[str, deque] = {
    "put": deque(maxlen=DEMO_CFG["per_task_max"]),
    "clean": deque(maxlen=DEMO_CFG["per_task_max"]),
    "heat": deque(maxlen=DEMO_CFG["per_task_max"]),
    "cool": deque(maxlen=DEMO_CFG["per_task_max"]),
    "puttwo": deque(maxlen=DEMO_CFG["per_task_max"]),
    "examine": deque(maxlen=DEMO_CFG["per_task_max"]),
}


def format_success_trajectory_as_demo(
    init_ob: str, task_line: str, steps: List[dict]
) -> str:
    """Format a successful trajectory as a compact ReAct-style demo."""
    lines: List[str] = []
    init_ob = (init_ob or "").strip()
    if init_ob:
        lines.append(init_ob)

    if task_line:
        t = task_line.strip()
        if t and (t not in init_ob):
            lines.append(t)

    for st in steps:
        if st["type"] == "think":
            th = (st.get("llm") or "").strip()
            if th:
                lines.append(f"> {th}")
                lines.append("OK.")
        else:
            ex = (st.get("exec") or "").strip()
            if ex:
                lines.append(f"> {ex}")
            ob = (st.get("obs") or "").strip()
            if ob:
                lines.append(ob)

    text = "\n".join(lines).strip()
    if len(text) > DEMO_CFG["max_chars_each"]:
        text = text[-DEMO_CFG["max_chars_each"] :]
    return text + "\n"


def build_demo_block(task_key: str) -> str:
    """Build an optional demo block from recent successful trajectories."""
    if (not DEMO_CFG["enable"]) or (task_key not in SUCCESS_DEMOS):
        return ""
    buf = list(SUCCESS_DEMOS[task_key])
    if not buf:
        return ""
    k = min(DEMO_CFG["inject_k"], len(buf))
    chosen = buf[-k:]
    block = "\n".join(chosen)
    return f"\n# Additional successful example (recent)\n{block}\n"


# =========================================================
# Online SFT data collection and micro-training
# =========================================================
SFT_CFG = {
    "enable": True,
    "history_k": 8,
    "max_obs_chars": 800,
    "max_init_obs_lines": 18,
    "include_actions_summary": False,
    "actions_summary_cap": 18,
    "drop_verify_steps": True,
    "drop_nothing_happens": False,
    "min_input_chars": 30,
    "buffer_max": 50000,
    "train_every_success": 10,
    "train_batch_size": 2,
    "train_grad_accum": 8,
    "train_steps": 80,
    "lr": 5e-5,
    "max_seq_len": 1024,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

SFT_DATA_PATH = SAVE_DIR / "sft_data.jsonl"
SFT_DATA_PATH.touch(exist_ok=True)
SFT_BUFFER: deque = deque(maxlen=SFT_CFG["buffer_max"])
SUCCESS_EPISODE_COUNT = 0


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n]


def _clean_obs_for_sft(ob: str) -> str:
    ob = process_ob(ob or "")
    if not ob:
        return ""
    if ob.strip().lower() == "ok.":
        return ""
    return _clip(ob, SFT_CFG["max_obs_chars"])


def build_sft_input(task_line: str, init_ob: str, history: List[Tuple[str, str]]) -> str:
    """Build SFT input in (TASK, INIT_OBS, HISTORY, NEXT_ACTION) format."""
    init_lines = (init_ob or "").splitlines()
    if len(init_lines) > SFT_CFG["max_init_obs_lines"]:
        init_ob = "\n".join(init_lines[: SFT_CFG["max_init_obs_lines"]])

    parts: List[str] = []
    if task_line:
        parts.append(f"TASK:\n{task_line.strip()}")
    if init_ob:
        parts.append(f"INIT_OBS:\n{init_ob.strip()}")

    if history:
        h_lines: List[str] = []
        for idx, (a, o) in enumerate(history, 1):
            a = (a or "").strip()
            o = (o or "").strip()
            if not a and not o:
                continue
            h_lines.append(f"{idx}) ACT: {a}")
            if o:
                h_lines.append(f"   OBS: {o}")
        if h_lines:
            parts.append("HISTORY:\n" + "\n".join(h_lines))

    parts.append("NEXT_ACTION:")
    return "\n\n".join(parts).strip()


def traj_to_sft_samples(
    traj: dict,
    init_ob: str,
    task_line: str,
    episode_name: str,
    split: str,
    task_key: str,
) -> List[dict]:
    """Convert a successful trajectory into multiple SFT training samples."""
    steps = traj.get("steps", [])
    samples: List[dict] = []

    act_obs_seq: List[Tuple[str, str, dict]] = []
    for st in steps:
        if st.get("type") != "act":
            continue
        exec_act = (st.get("exec") or "").strip()
        obs = _clean_obs_for_sft(st.get("obs") or "")
        if not exec_act:
            continue
        if SFT_CFG["drop_verify_steps"] and exec_act.strip().lower() == "verify_on_target":
            continue
        if SFT_CFG["drop_nothing_happens"] and ("nothing happens" in (obs or "").lower()):
            continue
        act_obs_seq.append((exec_act, obs, st))

    def infer_origin(st_any: dict) -> str:
        """Lightweight tag indicating whether the action came from actor, critic, or fallback."""
        llm_raw = (st_any.get("llm") or "").strip().lower()
        exec_raw = (st_any.get("exec") or "").strip().lower()
        if exec_raw == "verify_on_target":
            return "verify"
        if ("put " in llm_raw or "move " in llm_raw or "drop " in llm_raw) and (exec_raw != llm_raw):
            return "fallback"
        return "actor"

    K = SFT_CFG["history_k"]
    for t in range(len(act_obs_seq)):
        target_exec, _, st_any = act_obs_seq[t]

        start = max(0, t - K)
        hist_slice = act_obs_seq[start:t]
        history_pairs = [(a, o) for (a, o, _) in hist_slice]

        inp = build_sft_input(task_line=task_line, init_ob=init_ob, history=history_pairs)
        if len(inp) < SFT_CFG["min_input_chars"]:
            continue

        samples.append(
            {
                "task_key": task_key,
                "episode": episode_name,
                "split": split,
                "t": t,
                "origin": infer_origin(st_any),
                "input": inp,
                "target": target_exec.strip(),
            }
        )

    return samples


def append_sft_samples(samples: List[dict]) -> None:
    """Append SFT samples to on-disk dataset and in-memory buffer."""
    if not samples:
        return
    with open(SFT_DATA_PATH, "a", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            SFT_BUFFER.append(ex)


def _maybe_setup_lora(model):
    """Attach LoRA adapters if possible; otherwise return the original model."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        return model, False

    if hasattr(model, "peft_config"):
        return model, True

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=SFT_CFG["lora_r"],
        lora_alpha=SFT_CFG["lora_alpha"],
        lora_dropout=SFT_CFG["lora_dropout"],
        target_modules=SFT_CFG["lora_target_modules"],
        bias="none",
    )
    m = get_peft_model(model, cfg)
    m.print_trainable_parameters()
    return m, True


def _batch_tokenize(ex_list: List[dict]) -> Dict[str, torch.Tensor]:
    """Batch tokenizer for SFT inputs and labels with input masking."""
    texts: List[str] = []
    for ex in ex_list:
        inp = ex["input"].rstrip()
        tgt = ex["target"].strip()
        texts.append(inp + " " + tgt)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SFT_CFG["max_seq_len"],
    )

    labels = enc["input_ids"].clone()
    for i, ex in enumerate(ex_list):
        inp = ex["input"].rstrip()
        inp_ids = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            max_length=SFT_CFG["max_seq_len"],
        )["input_ids"][0]
        L_inp = int(inp_ids.shape[0])
        labels[i, :L_inp] = -100

    enc["labels"] = labels
    return {k: v.to(device) for k, v in enc.items()}


def micro_sft_train() -> None:
    """Small-scale online SFT update from the current success buffer."""
    if not SFT_CFG["enable"]:
        return
    if len(SFT_BUFFER) < 50:
        print(f"[SFT] Buffer too small ({len(SFT_BUFFER)}). Skip training.")
        return

    global model
    model, lora_ok = _maybe_setup_lora(model)
    if not lora_ok:
        print("[SFT] peft not installed. Skipping training (data is still collected).")
        return

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=SFT_CFG["lr"])

    data = list(SFT_BUFFER)
    steps = SFT_CFG["train_steps"]
    bs = SFT_CFG["train_batch_size"]
    ga = SFT_CFG["train_grad_accum"]

    print(
        f"[SFT] Micro-training start: steps={steps}, bs={bs}, grad_accum={ga}, buffer={len(data)}"
    )

    optim.zero_grad(set_to_none=True)
    for step in range(1, steps + 1):
        batch = random.sample(data, k=min(bs, len(data)))
        batch_enc = _batch_tokenize(batch)

        out = model(**batch_enc)
        loss = out.loss / ga
        loss.backward()

        if step % ga == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print(f"[SFT] step={step}/{steps} | loss={float(loss.item()) * ga:.4f}")

    model.eval()
    print("[SFT] Micro-training done.")


# =========================================================
# Main actor-critic interaction loop (single ALFWorld episode)
# =========================================================
def alfworld_run(
    prompt: str,
    to_print: bool = True,
    ob: str = "",
    info=None,
    episode_name: str = "",
    split: str = "",
    task_key: str = "",
) -> tuple:
    """
    Run a single ALFWorld episode with:
      - Actor (System-1) executing actions
      - Critic invoked selectively via ambiguity gating
    """
    task_line = ""
    for line in ob.splitlines():
        if line.lower().startswith("your task is to:"):
            task_line = line
            break

    init_prompt = prompt + ob + "\n>"
    prompt_ctx = ""

    if to_print:
        print("[ACTOR] Initial observation:\n" + ob)
        sys.stdout.flush()

    traj: Dict[str, Any] = {
        "episode_name": episode_name,
        "split": split,
        "task_key": task_key,
        "success": None,
        "total_steps": 0,
        "steps": [],
    }

    env_steps = 0
    curr_info = info
    last_critic_step = -999

    for i in range(1, 30):
        # 1) Actor: propose next action (or thought)
        llm_input = init_prompt + prompt_ctx
        action_txt = llm(llm_input, stop=["\n"], max_new_tokens=100).strip()

        # Thought-only step (internal reasoning)
        if action_txt.lower().startswith("think:"):
            if to_print:
                print(f"[ACTOR] Act {i}: {action_txt}\n[ACTOR] Obs {i}: OK.")
            prompt_ctx += f" {action_txt}\nOK.\n>"
            traj["steps"].append(
                {
                    "i": i,
                    "type": "think",
                    "llm": action_txt,
                    "exec": None,
                    "obs": "OK.",
                    "reward": None,
                    "done": False,
                }
            )
            continue

        # 2) Ambiguity-based gate for critic invocation
        admissible_for_gate: List[str] = []
        if curr_info is not None:
            admissible_for_gate = get_valid_actions(curr_info)

        is_ambiguous = False
        if admissible_for_gate:
            is_ambiguous = should_invoke_critic(
                prefix=llm_input,
                admissible=admissible_for_gate,
                step_idx=i,
                episode_name=episode_name,
                to_print=to_print,
            )

        # 3) Critic call (gated)
        cooldown_ok = (i - last_critic_step) >= GATE_CFG.get("cooldown", 0)
        call_critic = bool(is_ambiguous and cooldown_ok)

        if call_critic:
            last_critic_step = i

            # Ensure we have a valid state to query admissible actions from
            if curr_info is None:
                obs0, reward0, done0, curr_info = env.step(["look around"])
                env_steps += 1
                ob0 = process_ob(obs0[0])
                traj["steps"].append(
                    {
                        "i": i,
                        "type": "act",
                        "llm": "look around (auto-init by critic)",
                        "exec": "look around",
                        "obs": ob0,
                        "reward": reward0[0],
                        "done": done0[0],
                    }
                )
                if to_print:
                    print(f"[ACTOR] Auto-init (by critic): look around\n[ACTOR] Obs: {ob0}")

            admissible_now = get_valid_actions(curr_info)
            traj_snip = _make_trajectory_snippet(traj["steps"])
            advice_data = critic_step_advice(task_line, traj_snip, admissible_now)

            if to_print:
                print(f"[CRITIC] Triggered at step {i} (ambiguity gate).")
                print(f"[CRITIC] Proposed action: {advice_data.get('next_action')}")

            critic_action = (advice_data.get("next_action") or "").strip()
            if critic_action and critic_action.lower() != "continue":
                fixed_action = canonicalize_action(critic_action, admissible_now)
                if fixed_action:
                    action_txt = fixed_action
                    if to_print:
                        print(f"   ---> [CRITIC OVERRIDE] Executing: {action_txt}")

        # 4) Ensure current admissible actions
        if curr_info is None:
            obs, reward, done, curr_info = env.step(["look around"])
            env_steps += 1
            ob0 = process_ob(obs[0])
            traj["steps"].append(
                {
                    "i": i,
                    "type": "act",
                    "llm": "look around (auto-init)",
                    "exec": "look around",
                    "obs": ob0,
                    "reward": reward[0],
                    "done": done[0],
                }
            )
            if to_print:
                print(f"[ACTOR] Auto-init: look around\n[ACTOR] Obs: {ob0}")

        admissible = get_valid_actions(curr_info)

        # 5) Canonicalize actor/critic proposal into an admissible action
        act = canonicalize_action(action_txt, admissible)
        if act is None:
            t = re.sub(r"^[Aa]ction:\s*", "", action_txt).strip()
            act = canonicalize_action(t, admissible)
        if act is None:
            for pref in ("examine ", "open ", "look around", "inventory"):
                cand = [v for v in admissible if v.lower().startswith(pref)]
                if cand:
                    act = cand[0]
                    break
        if act is None:
            act = random.choice(admissible)

        # 6) Structured put/move/drop with fallback
        low = act.lower()
        if low.startswith(("put ", "move ", "drop ")):
            m_put = re.search(r"put\s+(.+?)\s+(?:on|in)\s+(.+)$", low)
            m_mov = re.search(r"move\s+(.+?)\s+to\s+(.+)$", low)
            if m_put or m_mov:
                if m_put:
                    obj, target = m_put.group(1), m_put.group(2)
                else:
                    obj, target = m_mov.group(1), m_mov.group(2)

                obs, reward, done, curr_info, ok, used, subtrace = put_with_fallback(
                    env, curr_info, obj=obj, target=target
                )
                env_steps += used

                for st in subtrace:
                    traj["steps"].append(
                        {
                            "i": i,
                            "type": "act",
                            "llm": action_txt,
                            "exec": st["exec"],
                            "obs": st["obs"],
                            "reward": st["reward"],
                            "done": st["done"],
                        }
                    )

                ob_str = process_ob(obs[0])
                if to_print:
                    print(f"[ACTOR] Act {i}: {act}\n[ACTOR] Obs {i}: {ob_str}")
                prompt_ctx += f" {act}\n{ob_str}\n>"

                if done[0]:
                    won = unpack_info(curr_info)["won"][0]
                    traj["success"] = bool(won)
                    traj["total_steps"] = env_steps
                    return won, traj, env_steps, task_line
                continue

        # 7) Default single environment step
        obs, reward, done, curr_info = env.step([act])
        env_steps += 1
        ob_str = process_ob(obs[0])

        if to_print:
            print(f"[ACTOR] Act {i}: {act}\n[ACTOR] Obs {i}: {ob_str}")

        prompt_ctx += f" {act}\n{ob_str}\n>"
        traj["steps"].append(
            {
                "i": i,
                "type": "act",
                "llm": action_txt,
                "exec": act,
                "obs": ob_str,
                "reward": reward[0],
                "done": done[0],
            }
        )

        if done[0]:
            won = unpack_info(curr_info)["won"][0]
            traj["success"] = bool(won)
            traj["total_steps"] = env_steps
            return won, traj, env_steps, task_line

    # Episode timeout
    won = 0
    traj["success"] = False
    traj["total_steps"] = env_steps
    return won, traj, env_steps, task_line


# =========================================================
# Benchmark loop
# =========================================================
prefixes = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}
cnts = [0] * 6
rs = [0] * 6

# Load base prompt examples
folder = "./prompts/"
prompt_file = "alfworld_prompts.json"
with open(os.path.join(folder, prompt_file), "r") as f:
    d = json.load(f)

print(f"Overall we have 134 games in split={split}")

for epi_idx in range(134):
    ob, info = env.reset()
    ob = "\n".join(ob[0].split("\n\n")[1:])
    name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
    print(name)

    r = 0
    traj = None
    steps_used = 0
    task_line = ""
    task_key = ""

    for j, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            task_key = v

            demo_block = build_demo_block(v)
            prompt = (
                "Interact with a household environment to solve a task. "
                "Here are two base examples.\n"
                + d[f"react_{v}_1"]
                + d[f"react_{v}_0"]
                + demo_block
                + "\nHere is the task.\n"
            )
            print(k, v, f"(demo buffer size={len(SUCCESS_DEMOS[v])})")

            r, traj, steps_used, task_line = alfworld_run(
                prompt,
                ob=ob,
                info=info,
                episode_name=name,
                split=split,
                task_key=v,
            )

            rs[j] += r
            cnts[j] += 1

            if r == 1:
                success_steps_list.append(steps_used)
                success_trajectories.append(traj)
            else:
                failure_trajectories.append(traj)
            break

    # On success: update demo buffer and SFT buffer, then optionally run micro SFT
    if r == 1 and traj is not None and task_key:
        # Demo buffer update
        if DEMO_CFG["enable"] and (task_key in SUCCESS_DEMOS):
            ex = format_success_trajectory_as_demo(
                init_ob=ob, task_line=task_line, steps=traj["steps"]
            )
            SUCCESS_DEMOS[task_key].append(ex)

        # SFT sample collection
        if SFT_CFG["enable"]:
            SUCCESS_EPISODE_COUNT += 1
            samples = traj_to_sft_samples(
                traj=traj,
                init_ob=ob,
                task_line=task_line,
                episode_name=name,
                split=split,
                task_key=task_key,
            )
            append_sft_samples(samples)
            print(
                f"[SFT] Collected samples: +{len(samples)} | "
                f"buffer={len(SFT_BUFFER)} | success_eps={SUCCESS_EPISODE_COUNT}"
            )

            if SUCCESS_EPISODE_COUNT % SFT_CFG["train_every_success"] == 0:
                micro_sft_train()

    total_cnt = max(1, sum(cnts))
    avg_success_len = (
        sum(success_steps_list) / len(success_steps_list) if success_steps_list else 0.0
    )
    print(
        epi_idx + 1,
        "r",
        r,
        "rs",
        rs,
        "cnts",
        cnts,
        "AvgScore",
        sum(rs) / total_cnt,
        "AvgLen",
        avg_success_len,
    )
    print("------------\n")

# =========================================================
# Save results
# =========================================================
save_all_json()
print(f"model_name={MODEL_NAME}")
print(f"[SAVE] success.json, failure.json, decision_analysis.json saved in {SAVE_DIR}")
print(f"[SAVE] SFT data saved at {SFT_DATA_PATH}")
print(f"Total Prompt Tokens: {TOTAL_PROMPT_TOKENS}")
print(f"Total Generated Tokens: {TOTAL_GEN_TOKENS}")
