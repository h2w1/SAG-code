import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # change this to "0" etc. as needed

import sys
import json
import random
from collections import deque

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.generation.stopping_criteria import (
    StoppingCriteria, StoppingCriteriaList,
)

from peft import LoraConfig, get_peft_model, PeftModel

import gym
import babyai_text


# =========================
# Global configuration
# =========================

ACTION_SPACE = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]

PROMPT_HEADER_REACT = (
    "You are an expert, logical agent in a grid world. "
    "Your ONLY task is to output your reasoning and a single action in a specific format. "
    "You MUST follow this format EXACTLY. Do NOT add any extra words, explanations, or conversational text. "
    "ONLY provide the 'Thought' and 'Action' lines.\n"
    "Use this exact format:\n"
    "Thought: <your brief reasoning about the next best step>\n"
    "Action: <one action from the Valid actions list>"
)

MAX_CONTEXT_TOKENS = 4096
HISTORY_TURNS = 5
THOUGHT_TOKENS = 48
ACTION_TOKENS = 16
NUM_BEAMS = 1
DO_SAMPLE = False

# Simple loop-heuristic (not part of the core SAG gate)
USE_LOOP_INTERVENTION = True
LOOP_DETECTION_THRESHOLD = 3  # threshold on revisiting the same observation line

# SAG-style ambiguity gate configuration
USE_SAG_GATE = True
SAG_ENTROPY_THRESH = 0.9
SAG_MARGIN_THRESH = 1.0
SAG_UNCERT_STEPS = 3
SAG_MIN_STEP = 6

# Success-demonstration (few-shot) configuration
USE_SUCCESS_DEMONSTRATIONS = True
DEMO_MAX_EXAMPLES = 2
DEMO_MAX_STEPS_PER_EXAMPLE = 12
DEMO_MAX_TOKENS_BUDGET = 900
DEMO_INCLUDE_THOUGHT = True

# Online LoRA SFT (online self-improvement) configuration
USE_ONLINE_LORA_SFT = True
ONLINE_SFT_TRIGGER_SUCCESS_EPISODES = 10   # run SFT after this many successful episodes
ONLINE_SFT_MAX_STEPS_PER_EP = 20          # max steps per episode used for SFT data
ONLINE_SFT_ONLY_ACTOR_ACTION = True       # use only actor actions (no critic actions) for SFT
ONLINE_SFT_OUTPUT_DIR = "online_lora_sft" # Trainer output directory (no checkpoints saved)

# Global token counters (for token-efficiency reporting)
TOTAL_PROMPT_TOKENS = 0
TOTAL_GEN_TOKENS = 0


# =========================
# Utility functions
# =========================

def get_valid_actions(obs_descs):
    """Compute admissible actions based on textual observations."""
    if_obj_forward = if_wall_forward = if_door_forward = if_hold_obj = False
    for o in obs_descs:
        if 'and' not in o and o.startswith('You see') and o.endswith('1 step forward'):
            if 'wall' in o:
                if_wall_forward = True
            elif 'door' in o:
                if_door_forward = True
            else:
                if_obj_forward = True
        if 'carry' in o:
            if_hold_obj = True
    valid = ['turn left', 'turn right']
    if if_obj_forward and not if_hold_obj:
        valid.append('pick up')
    if not if_obj_forward and not if_wall_forward and not if_door_forward:
        valid.append('go forward')
        if if_hold_obj:
            valid.append('drop')
    if if_door_forward:
        valid.append('toggle')
    return valid if valid else ['turn left', 'turn right']


class MyStopping(StoppingCriteria):
    """Stop generation when a given string appears after the starting position."""
    def __init__(self, start_length, tokenizer, stop_str):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_str = stop_str
        self.start_length = start_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        decoded_text = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        return any(self.stop_str in s for s in decoded_text)


def get_stopping(start_length, stop_strs, tokenizer):
    return StoppingCriteriaList([MyStopping(start_length, tokenizer, s) for s in stop_strs])


def clamp_text_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text from the left to fit into a token budget."""
    if not text or max_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True).strip()


def build_react_preamble() -> str:
    """Base system prompt for the actor."""
    return PROMPT_HEADER_REACT


def normalize_action(pred: str, valid_actions: list) -> str:
    """Normalize raw action text into one of the valid actions."""
    if "action:" in pred.lower():
        pred = pred.lower().split("action:")[1]
    cand = pred.strip().lower().rstrip(".")
    for head in ["action:", "action -", "action-", "action ", "action"]:
        if cand.startswith(head):
            cand = cand[len(head):].strip()
            break
    if cand in valid_actions:
        return cand
    for a in valid_actions:
        if a in cand:
            return a
    return random.choice(valid_actions)


def trim_history_lines(history_lines: list, max_turns: int) -> list:
    """Keep at most max_turns dialogue turns (system/user/assistant pairs)."""
    if not max_turns or max_turns <= 0:
        return []
    max_items = max_turns * 2
    return history_lines[-max_items:] if len(history_lines) > max_items else history_lines


# =========================
# Formatting successful trajectories as demonstrations
# =========================

def format_success_trajectory_for_demos(ep_log: dict,
                                        max_steps: int = 12,
                                        include_thought: bool = True) -> str:
    """Format a successful trajectory into a compact demonstration snippet."""
    goal = ep_log.get("goal", "")
    init_obs = ep_log.get("initial_observation", [])
    steps = ep_log.get("steps", [])

    steps_show = steps[:max_steps]

    lines = []
    lines.append("### Successful Example")
    lines.append(f"Goal: {goal}")
    if init_obs:
        lines.append("Initial Observation: " + ", ".join(init_obs))

    for s in steps_show:
        obs_line = s.get("obs_line", "")
        act = s.get("action", "")
        th = s.get("thought", "")
        lines.append(f"Observation: {obs_line}")
        if include_thought and th:
            lines.append(f"Thought: {th}")
        lines.append(f"Action: {act}")

    lines.append("Result: Success")
    return "\n".join(lines).strip()


def build_system_prompt_with_demos(base_preamble: str,
                                   tokenizer,
                                   success_exemplars: deque) -> str:
    """Augment the system prompt with a small set of successful demonstrations."""
    if not USE_SUCCESS_DEMONSTRATIONS or not success_exemplars:
        return base_preamble

    examples_text = []
    examples_text.append(
        "Below are a few successful demonstrations. "
        "Use them as guidance. Follow the required Thought/Action format exactly."
    )
    for ex in list(success_exemplars):
        examples_text.append(ex)

    examples_block = "\n\n".join(examples_text)
    examples_block = clamp_text_tokens(examples_block, tokenizer, DEMO_MAX_TOKENS_BUDGET)

    return base_preamble + "\n\n" + examples_block


# =========================
# LLM call: ReAct-style Thought + Action for the actor
# =========================

@torch.inference_mode()
def generate_react_thought_action(model, tokenizer, messages, max_new_tokens):
    """Generate a Thought and Action continuation given chat messages."""
    global TOTAL_PROMPT_TOKENS, TOTAL_GEN_TOKENS

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = clamp_text_tokens(prompt, tokenizer, MAX_CONTEXT_TOKENS)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopping_criteria = get_stopping(inputs.input_ids.shape[-1], ['\n\n'], tokenizer)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=DO_SAMPLE,
        num_beams=NUM_BEAMS,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id
    )

    prompt_tokens = int(inputs.input_ids.shape[-1])
    gen_tokens = int(outputs.shape[-1] - inputs.input_ids.shape[-1])
    TOTAL_PROMPT_TOKENS += prompt_tokens
    TOTAL_GEN_TOKENS += gen_tokens

    response = outputs[0][inputs.input_ids.shape[-1]:]
    text = tokenizer.decode(response, skip_special_tokens=True).strip()
    return text


def react_step(model, tokenizer, system_prompt, current_observation, chat_history):
    """Single ReAct step for the actor: returns (thought, raw_action, user_msg, assistant_msg)."""
    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append({"role": "user", "content": current_observation})

    raw_output = generate_react_thought_action(
        model, tokenizer, messages, THOUGHT_TOKENS + ACTION_TOKENS
    )

    thought, action_text = "", ""
    lines = raw_output.split('\n')
    for line in lines:
        if line.lower().startswith("thought:"):
            thought = line[len("thought:"):].strip()
        elif line.lower().startswith("action:"):
            action_text = line[len("action:"):].strip()
    if not thought:
        thought = "No thought generated."
    if not action_text:
        action_text = raw_output

    return thought, action_text, {"role": "user", "content": current_observation}, {"role": "assistant", "content": raw_output}


# =========================
# Ambiguity signals (entropy + margin) for SAG gate
# =========================

@torch.inference_mode()
def compute_ambiguity_signals(model, tokenizer, messages, valid_actions):
    """
    Compute actor-side ambiguity signals for the SAG gate.

    For each admissible action, we score a short completion "Action: <a>",
    compute a normalized distribution over actions, and derive:
      - entropy over the action distribution
      - top-1 vs. top-2 margin (log-prob difference)
    """
    if not USE_SAG_GATE:
        return None
    if not valid_actions:
        return None

    base_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base_prompt = clamp_text_tokens(base_prompt, tokenizer, MAX_CONTEXT_TOKENS)
    base_inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
    base_ids = base_inputs["input_ids"][0]
    base_len = base_ids.shape[0]

    logps = []
    for a in valid_actions:
        full_text = base_prompt + f"\nAction: {a}"
        enc = tokenizer(full_text, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]  # [L, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)

        tokens = input_ids[0]
        start = base_len - 1
        end = tokens.shape[0] - 1
        lp = 0.0
        for i in range(start, end):
            next_tok = tokens[i + 1]
            lp += float(log_probs[i, next_tok])
        logps.append(lp)

    logps_tensor = torch.tensor(logps, dtype=torch.float32)
    probs = torch.softmax(logps_tensor, dim=-1)

    entropy = -float(torch.sum(probs * torch.log(probs + 1e-12)))

    sorted_idx = torch.argsort(logps_tensor, descending=True)
    top1_idx = int(sorted_idx[0])
    top2_idx = int(sorted_idx[1]) if len(valid_actions) > 1 else top1_idx
    top1_logp = float(logps_tensor[top1_idx])
    top2_logp = float(logps_tensor[top2_idx])
    top1_action = valid_actions[top1_idx]
    top2_action = valid_actions[top2_idx]
    top2margin = top1_logp - top2_logp if len(valid_actions) > 1 else float("inf")

    return {
        "entropy": entropy,
        "top2margin": top2margin,
        "top1_action": top1_action,
        "top1_logp": top1_logp,
        "top2_action": top2_action,
        "top2_logp": top2_logp,
    }


# =========================
# Critic LLM (plans / corrective actions)
# =========================

CRITIC_SYSTEM_PROMPT = (
    "You are a critic helping an actor agent navigate a small grid world.\n"
    "The actor may be stuck in a loop or highly uncertain.\n"
    "You will be given the current goal, observation, and the list of valid actions.\n"
    "Your task is to propose a SHORT sequence of 1 to 3 actions as a corrective plan.\n"
    "Only use actions from the valid actions list.\n"
    "Output format (MUST follow exactly):\n"
    "Plan: action1, action2, action3\n"
    "If you give fewer actions, just stop earlier, e.g. 'Plan: action1, action2'."
)

CRITIC_MAX_NEW_TOKENS = 32


@torch.inference_mode()
def get_critic_plan(model, tokenizer, goal, obs_line, valid_actions, reason_str):
    """
    Query the critic for a short corrective plan (1-3 actions) given a high-risk state.
    """
    if not valid_actions:
        return ["turn left"], "Plan: turn left"

    user_content = (
        f"The actor seems stuck or highly uncertain.\n"
        f"Reason: {reason_str}\n\n"
        f"Goal: {goal}\n"
        f"Current observation: {obs_line}\n"
        f"Valid actions: {', '.join(valid_actions)}\n\n"
        f"Remember: Output exactly one line starting with 'Plan:' followed by 1 to 3 actions separated by commas."
    )

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = clamp_text_tokens(prompt, tokenizer, MAX_CONTEXT_TOKENS)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stopping_criteria = get_stopping(inputs.input_ids.shape[-1], ['\n\n'], tokenizer)

    outputs = model.generate(
        **inputs,
        max_new_tokens=CRITIC_MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        num_beams=NUM_BEAMS,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id
    )

    response = outputs[0][inputs.input_ids.shape[-1]:]
    raw_text = tokenizer.decode(response, skip_special_tokens=True).strip()

    plan_text = raw_text
    if "plan:" in plan_text.lower():
        plan_text = plan_text.lower().split("plan:")[1].strip()

    cand_actions = [p.strip() for p in plan_text.split(",") if p.strip()]

    plan_actions = []
    for ca in cand_actions:
        norm = normalize_action(ca, valid_actions)
        if norm and norm in valid_actions:
            if not plan_actions or plan_actions[-1] != norm:
                plan_actions.append(norm)
        if len(plan_actions) >= 3:
            break

    if not plan_actions:
        fallback = []
        if 'turn right' in valid_actions:
            fallback.append('turn right')
        elif 'turn left' in valid_actions:
            fallback.append('turn left')
        if 'go forward' in valid_actions:
            fallback.append('go forward')
        if fallback:
            plan_actions = fallback
            raw_text = "Plan: " + ", ".join(fallback)
        else:
            plan_actions = [valid_actions[0]]
            raw_text = f"Plan: {valid_actions[0]}"

    return plan_actions, raw_text


# =========================
# Online LoRA SFT: Dataset & Trainer
# =========================

class TrajectorySFTDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        """
        examples: list of dicts
          {
            "input_text": "...",
            "target_text": "Thought: ...\nAction: ..."
          }
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_text = ex["input_text"]
        target_text = ex["target_text"]

        # For causal LM: concatenate input and target, then mask out the input part with -100.
        full_text = input_text + target_text

        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized_full["input_ids"][0]
        attn_mask = tokenized_full["attention_mask"][0]

        # Compute the length of the input segment to mask its labels.
        input_ids_only = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0]
        input_len = input_ids_only.shape[0]

        labels = input_ids.clone()
        labels[:input_len] = -100  # ignore input part in the loss

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


def build_sft_examples_from_successes(success_episodes_buffer,
                                      max_steps_per_ep=20,
                                      only_actor=True):
    """
    Build SFT (input_text, target_text) pairs from successful episodes.

    Each example corresponds to a (context, actor decision) pair, potentially
    including critic-corrected actions depending on configuration.
    """
    examples = []
    for ep in success_episodes_buffer:
        goal = ep.get("goal", "")
        steps = ep.get("steps", [])[:max_steps_per_ep]
        for s in steps:
            if only_actor and s.get("action_source") != "Actor":
                continue
            obs_line = s.get("obs_line", "")
            val_acts = s.get("valid_actions", [])
            thought = s.get("thought", "")
            action = s.get("action", "")

            input_text = (
                PROMPT_HEADER_REACT
                + "\n\n"
                + f"Goal: {goal}\n"
                + f"Observation: {obs_line}\n"
                + f"Valid actions: {', '.join(val_acts)}\n"
            )
            target_text = f"Thought: {thought}\nAction: {action}\n"

            examples.append({
                "input_text": input_text,
                "target_text": target_text,
            })
    return examples


def ensure_lora_wrapped(model):
    """
    Wrap a base causal LM with a LoRA adapter if not already wrapped.

    This is an implementation detail for efficient online self-improvement.
    """
    if isinstance(model, PeftModel):
        return model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def run_online_lora_sft(model, tokenizer, success_episodes_buffer, run_idx: int):
    """
    Run one round of online LoRA SFT using the buffer of successful episodes.
    The model is updated in-place.
    """
    if not USE_ONLINE_LORA_SFT:
        return model

    print(f"\n===== Online LoRA SFT Run #{run_idx} start =====")
    # 1) Ensure model is LoRA-wrapped
    model = ensure_lora_wrapped(model)

    # 2) Build SFT dataset
    examples = build_sft_examples_from_successes(
        success_episodes_buffer,
        max_steps_per_ep=ONLINE_SFT_MAX_STEPS_PER_EP,
        only_actor=ONLINE_SFT_ONLY_ACTOR_ACTION,
    )
    print(f"Online SFT example count (step-level): {len(examples)}")

    if len(examples) == 0:
        print("No examples available. Skipping SFT.")
        return model

    dataset = TrajectorySFTDataset(examples, tokenizer, max_length=512)

    # 3) TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=os.path.join(ONLINE_SFT_OUTPUT_DIR, f"run_{run_idx}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1.0,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=True,
        optim="adamw_torch",
    )

    def data_collator(features):
        """Pad variable-length sequences for Trainer."""
        batch = {}
        keys = features[0].keys()
        for k in keys:
            tensors = [f[k] for f in features]
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                tensors,
                batch_first=True,
                padding_value=tokenizer.pad_token_id if k != "labels" else -100,
            )
        return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    model.train()
    trainer.train()
    model.eval()

    print(f"===== Online LoRA SFT Run #{run_idx} end =====\n")
    return model


# =========================
# Main ReAct loop (SAG gate + critic + logging + demos + online SFT)
# =========================

def run_react(model, tokenizer, env, task_type, start_idx, num_task, preamble, seed):
    results_summary = {
        "task_type": task_type,
        "completed_episodes": 0,
        "success_rate": 0.0,
        "avg_reward": 0.0,  # reward is not heavily used here but kept for completeness
        "avg_steps_to_success": 0.0,
        "steps_to_success": [],
        "success_trajectories": [],
        "failure_trajectories": []
    }

    # Buffer of compact success demonstrations for few-shot prompting
    success_exemplars = deque(maxlen=DEMO_MAX_EXAMPLES)

    # Buffer of successful episodes for online SFT and a counter of SFT runs
    success_episodes_buffer = []
    online_sft_run_count = 0

    all_episode_logs = []

    for i in range(num_task):
        episode_num = start_idx + i
        obs, done = env.reset(), False
        goal = obs[0]['mission']
        results_summary["completed_episodes"] += 1

        print(f"\n--- Starting Episode {results_summary['completed_episodes']}/{num_task} (Seed: {seed + episode_num}) ---")
        descs = obs[-1]['descriptions']
        print('Goal of the agent: ' + goal, '\nInitial Observation: ' + ', '.join(descs))
        sys.stdout.flush()

        chat_history = []
        max_steps = 30

        # Simple loop heuristic
        loop_detector = {}
        # Queue of critic-suggested actions (when a corrective plan has length > 1)
        critic_action_queue = deque()
        # Streak of high-ambiguity states for the SAG gate
        sag_high_streak = 0
        # States where the critic has already intervened
        critic_applied_states = set()

        episode_log = {
            "episode_index": episode_num,
            "seed": seed + episode_num,
            "goal": goal,
            "initial_observation": list(descs),
            "steps": [],
            "success": None,
            "final_reward": None,
            "final_steps": None,
            "demo_examples_used": list(success_exemplars),
        }

        reward = 0.0

        for step in range(max_steps):
            valid_actions = get_valid_actions(descs)
            obs_line = ", ".join(descs)

            # Update loop-heuristic counter
            loop_detector[obs_line] = loop_detector.get(obs_line, 0) + 1
            loop_count = loop_detector[obs_line]

            ambiguity_stats = None
            thought = ""
            raw_action = ""
            action_source = "Actor"

            # System prompt with optional success demonstrations
            system_prompt = build_system_prompt_with_demos(preamble, tokenizer, success_exemplars)

            if critic_action_queue:
                # Follow remaining actions from the critic's previously suggested plan
                action = critic_action_queue.popleft()
                raw_action = action
                thought = f"[Executing critic's corrective plan action: {action}]"
                action_source = "Critic"
            else:
                loop_warning = ""
                if loop_count > 1:
                    loop_warning = (
                        f"Warning: You have visited this exact state {loop_count} times. "
                        f"You might be in a loop. Try a different action than before.\n"
                    )

                current_observation_prompt = (
                    f"{loop_warning}"
                    f"Goal: {goal}\n"
                    f"Observation: {obs_line}\n"
                    f"Valid actions: {', '.join(valid_actions)}"
                )

                policy_messages = (
                    [{"role": "system", "content": system_prompt}] +
                    chat_history +
                    [{"role": "user", "content": current_observation_prompt}]
                )
                ambiguity_stats = compute_ambiguity_signals(model, tokenizer, policy_messages, valid_actions)

                if ambiguity_stats is not None:
                    print(
                        f"[Ambiguity] entropy={ambiguity_stats['entropy']:.6f}, "
                        f"top2margin={ambiguity_stats['top2margin']:.6f}, "
                        f"top1={ambiguity_stats['top1_action']} (logp={ambiguity_stats['top1_logp']:.4f}), "
                        f"top2={ambiguity_stats['top2_action']} (logp={ambiguity_stats['top2_logp']:.4f})"
                    )

                    if (
                        ambiguity_stats["entropy"] > SAG_ENTROPY_THRESH
                        and ambiguity_stats["top2margin"] < SAG_MARGIN_THRESH
                    ):
                        sag_high_streak += 1
                    else:
                        sag_high_streak = 0
                else:
                    sag_high_streak = 0

                need_critic = False
                reason_str = ""

                # Loop-based heuristic for critic invocation
                if USE_LOOP_INTERVENTION and loop_count >= LOOP_DETECTION_THRESHOLD:
                    if obs_line not in critic_applied_states:
                        need_critic = True
                        reason_str = f"visited the same state {loop_count} times (loop)"
                    else:
                        need_critic = False

                # SAG-style ambiguity gate combined with loop signal
                if (
                    not need_critic
                    and USE_SAG_GATE
                    and ambiguity_stats is not None
                    and (step + 1) >= SAG_MIN_STEP
                    and sag_high_streak >= SAG_UNCERT_STEPS
                    and loop_count >= 2
                    and obs_line not in critic_applied_states
                ):
                    need_critic = True
                    reason_str = (
                        f"high action uncertainty for {sag_high_streak} steps "
                        f"(entropy>{SAG_ENTROPY_THRESH}, margin<{SAG_MARGIN_THRESH}) "
                        f"and revisited state {loop_count} times"
                    )

                if need_critic:
                    print(f"  -> CRITIC INTERVENTION triggered. reason: {reason_str}")
                    critic_plan, critic_raw = get_critic_plan(
                        model=model,
                        tokenizer=tokenizer,
                        goal=goal,
                        obs_line=obs_line,
                        valid_actions=valid_actions,
                        reason_str=reason_str,
                    )
                    print(f"  -> Critic plan (LLM): {critic_plan}")
                    print(f"     Critic raw output: {critic_raw}")

                    critic_action_queue.extend(critic_plan)
                    action = critic_action_queue.popleft()
                    raw_action = action
                    thought = f"[Starting critic's corrective plan: {action}]"
                    action_source = "Critic"

                    critic_applied_states.add(obs_line)
                    sag_high_streak = 0
                else:
                    # Regular actor step with truncated history
                    chat_history = trim_history_lines(chat_history, HISTORY_TURNS)
                    thought, raw_action, user_msg, assistant_msg = react_step(
                        model, tokenizer, system_prompt, current_observation_prompt, chat_history
                    )
                    chat_history.extend([user_msg, assistant_msg])
                    action_source = "Actor"

            action = normalize_action(raw_action, valid_actions)
            a_id = ACTION_SPACE.index(action)

            _, reward, done, new_obs = env.step(a_id)
            descs = new_obs['descriptions']

            print(f"****** Step {step+1}:")
            print(f"Thought: {thought}")
            print(f"Action: {action} (Source: {action_source}, Raw: {raw_action})")
            print(f"Observation: {', '.join(descs)}")
            print("******")
            sys.stdout.flush()

            step_log = {
                "step": step + 1,
                "obs_line": obs_line,
                "observations": list(descs),
                "valid_actions": list(valid_actions),
                "action": action,
                "raw_action": raw_action,
                "action_source": action_source,
                "thought": thought,
                "loop_count": loop_count,
                "ambiguity_stats": ambiguity_stats,
                "num_demo_examples": len(success_exemplars),
            }
            episode_log["steps"].append(step_log)

            if done:
                break

        final_step = step + 1
        success = 1 if (done and reward > 0) else 0

        if success:
            results_summary["steps_to_success"].append(final_step)

        successes = len(results_summary["steps_to_success"])
        current_sr = successes / results_summary["completed_episodes"]
        print(f'Episode Result -> Success: {bool(success)}, Steps: {final_step}')
        print(f'Progress: {successes}/{results_summary["completed_episodes"]} success ({current_sr:.2%})')
        print('-----------------------------------\n')

        episode_log["success"] = bool(success)
        episode_log["final_reward"] = float(reward)
        episode_log["final_steps"] = final_step

        # Update demonstration buffer from successful trajectories
        if USE_SUCCESS_DEMONSTRATIONS and success:
            exemplar_text = format_success_trajectory_for_demos(
                ep_log=episode_log,
                max_steps=DEMO_MAX_STEPS_PER_EXAMPLE,
                include_thought=DEMO_INCLUDE_THOUGHT
            )
            success_exemplars.append(exemplar_text)

        # Update online SFT buffer and trigger online self-improvement if needed
        if USE_ONLINE_LORA_SFT and success:
            success_episodes_buffer.append(episode_log)
            # Run SFT every ONLINE_SFT_TRIGGER_SUCCESS_EPISODES successful episodes
            if len(success_episodes_buffer) % ONLINE_SFT_TRIGGER_SUCCESS_EPISODES == 0:
                online_sft_run_count += 1
                model = run_online_lora_sft(
                    model, tokenizer, success_episodes_buffer, run_idx=online_sft_run_count
                )

        all_episode_logs.append(episode_log)

    total_completed = results_summary["completed_episodes"]
    if total_completed > 0:
        results_summary["success_rate"] = len(results_summary["steps_to_success"]) / total_completed

    if results_summary["steps_to_success"]:
        results_summary["avg_steps_to_success"] = (
            sum(results_summary["steps_to_success"]) / len(results_summary["steps_to_success"])
        )
    else:
        results_summary["avg_steps_to_success"] = 0.0

    print('===== SUMMARY =====')
    print(f'Success Rate: {results_summary["success_rate"]:.2%}')
    print(f'Avg steps to success (successful episodes only): {results_summary["avg_steps_to_success"]:.2f}')

    global TOTAL_PROMPT_TOKENS, TOTAL_GEN_TOKENS
    total_tokens = TOTAL_PROMPT_TOKENS + TOTAL_GEN_TOKENS
    print('===== TOKEN STATS =====')
    print(f'Total prompt tokens: {TOTAL_PROMPT_TOKENS}')
    print(f'Total generated tokens: {TOTAL_GEN_TOKENS}')
    print(f'Total tokens: {total_tokens}')
    if total_completed > 0:
        avg_prompt = TOTAL_PROMPT_TOKENS / total_completed
        avg_gen = TOTAL_GEN_TOKENS / total_completed
        avg_total = total_tokens / total_completed
        print(f'Avg prompt tokens per episode: {avg_prompt:.2f}')
        print(f'Avg generated tokens per episode: {avg_gen:.2f}')
        print(f'Avg total tokens per episode: {avg_total:.2f}')

    return results_summary, all_episode_logs, model


# =========================
# Main: model/env loading & execution
# =========================

if __name__ == "__main__":
    model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
    # Alternative examples:
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"

    task_type, num_task, start_idx = "goto", 50, 1
    model_short = model_name_or_path.split('/')[-1]
    RESULTS_FILE_PATH = f"react_final_{model_short}_{task_type}.json"
    LOG_FILE_PATH = f"react_log_{model_short}_{task_type}.json"

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Loading model: {model_name_or_path} ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side='left'
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.eval()
    print("Models loaded successfully.")

    env = gym.make("BabyAI-GoToLocal-v0")
    preamble = build_react_preamble()
    print("Environment and preamble are ready.")

    results_data, run_logs, model = run_react(
        model, tokenizer, env, task_type, start_idx, num_task, preamble, SEED
    )

    def convert_numpy(obj):
        if isinstance(obj, np.number):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    try:
        with open(RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False, default=convert_numpy)
        print(f"\nResults successfully saved to {RESULTS_FILE_PATH}")
    except Exception as e:
        print(f"\nError saving results to file: {e}")

    try:
        with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(run_logs, f, indent=2, ensure_ascii=False, default=convert_numpy)
        print(f"Detailed run logs successfully saved to {LOG_FILE_PATH}")
    except Exception as e:
        print(f"\nError saving run logs to file: {e}")
