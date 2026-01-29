import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import sys
import re
import requests
import torch
from bs4 import BeautifulSoup
from bs4.element import Comment
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

import random
import json
import math
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from pathlib import Path

# =========================
# 전역 시드/결정론 설정
# =========================
SEED = 42
def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(SEED)

# ✅ 토큰 집계를 위한 전역 변수
TOTAL_PROMPT_TOKENS = 0
TOTAL_GEN_TOKENS = 0

def _add_tokens(prompt_tokens: int, gen_tokens: int):
    global TOTAL_PROMPT_TOKENS, TOTAL_GEN_TOKENS
    TOTAL_PROMPT_TOKENS += int(prompt_tokens)
    TOTAL_GEN_TOKENS += int(gen_tokens)

# =========================
# CLI 인자
# =========================
parser = argparse.ArgumentParser()
parser.add_argument('--port', required=True, type=int)
parser.add_argument('--model_name_or_path', required=False, type=str)
parser.add_argument('--add_special_tokens', required=False, type=bool, default=False)
args = parser.parse_args()

WEBSHOP_URL = f"http://localhost:{args.port}"
MODEL_NAME = args.model_name_or_path

# =========================
# 모델 로드
# =========================
use_bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
)
print(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.eval()
DEVICE = next(model.parameters()).device  # 안전하게 실제 파라미터 디바이스 가져오기

# =========================
# LLM 호출
# =========================
def llm(prompt: str, stop: List[str] = ["\n"], max_new_tokens: int = 100):
    """
    공통 LLM 호출 유틸
    return: (text, input_tokens, output_tokens)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    prompt_tokens = int(inputs.input_ids.shape[1])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[:, inputs.input_ids.shape[1]:]
    gen_tokens = int(gen_ids.shape[1])
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    if stop:
        cut = len(text)
        for s in stop:
            pos = text.find(s)
            if pos != -1:
                cut = min(cut, pos)
        text = text[:cut]

    _add_tokens(prompt_tokens, gen_tokens)
    return text.strip(), prompt_tokens, gen_tokens

# =========================================================
# 🔵 SAG-style ambiguity 계산 유틸
# =========================================================

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

def extract_buttons_from_observation(obs: str):
    seen = set()
    btns = []
    for m in BRACKET_RE.finditer(obs):
        b = m.group(1).strip()
        if not b:
            continue
        if b not in seen:
            seen.add(b)
            btns.append(b)
    return btns

def admissible_click_actions(env, session_id: str, observation: str):
    page_type = env.sessions.get(session_id, {}).get("page_type", None)
    if page_type in (None, "init", "end"):
        return []

    buttons = extract_buttons_from_observation(observation)

    if page_type == "search":
        buttons = [b for b in buttons if b not in ("Next >", "< Prev")]

    return [f"click[{b}]" for b in buttons]

MAX_SCORE_TOKENS = 1024

def batch_logprob_of_candidates(prompt: str, candidates):
    if len(candidates) == 0:
        return []

    cand_texts = [" " + c for c in candidates]

    prompt_ids_full = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    prompt_ids = prompt_ids_full[-MAX_SCORE_TOKENS:] if prompt_ids_full.shape[0] > MAX_SCORE_TOKENS else prompt_ids_full
    prompt_ids = prompt_ids.to(DEVICE)
    prompt_len = prompt_ids.shape[0]

    scores = []
    for ct in cand_texts:
        cand_ids = tokenizer(ct, return_tensors="pt", add_special_tokens=False).input_ids[0].to(DEVICE)
        clen = cand_ids.shape[0]

        input_ids = torch.cat([prompt_ids, cand_ids], dim=0).unsqueeze(0)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits
            logprobs = F.log_softmax(logits, dim=-1)

        total = 0.0
        for j in range(clen):
            token_index = prompt_len + j
            prev_index = token_index - 1
            if prev_index < 0:
                continue
            tok = cand_ids[j].item()
            total += logprobs[0, prev_index, tok].item()

        scores.append(total)

        del input_ids, attn, out, logits, logprobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scores

def ambiguity_signals_from_candidates(prompt: str, candidates):
    K = len(candidates)
    if K < 2:
        return None

    lps = batch_logprob_of_candidates(prompt, candidates)
    lp_t = torch.tensor(lps, dtype=torch.float64)
    q = torch.softmax(lp_t, dim=0)

    eps = 1e-12
    H = -(q * torch.log(q + eps)).sum().item()
    Hbar = H / math.log(K)

    q_sorted, idx = torch.sort(q, descending=True)
    margin = (torch.log(q_sorted[0] + eps) - torch.log(q_sorted[1] + eps)).item()

    topk = []
    for r in range(min(5, K)):
        topk.append((candidates[idx[r].item()], q_sorted[r].item()))

    return {
        "K": K,
        "Hbar": float(Hbar),
        "margin": float(margin),
        "topk": topk,
    }

# =========================================================
# 크리틱 트리거 규칙
# =========================================================
CRITIC_CFG = {
    "enable": True,
    "hbar_threshold": 0.30,
    "margin_threshold": 1.0,
}

def should_call_critic(amb):
    if not CRITIC_CFG["enable"]:
        return False
    if amb is None:
        return False
    if amb["Hbar"] >= CRITIC_CFG["hbar_threshold"] or amb["margin"] <= CRITIC_CFG["margin_threshold"]:
        return True
    return False

def critic_suggest_action(amb):
    if amb is None or not amb["topk"]:
        return None
    return amb["topk"][0][0]

# =========================================================
# WebShop용 LLM 기반 Critic (짧은 프롬프트)
# =========================================================
WEB_CRITIC_SYS_PROMPT = """You are a short, decisive assistant for a WebShop shopping agent.

Task:
- Given the user instruction, recent history, and allowed actions, choose ONE best next action.

Guidelines:
- Move toward satisfying all constraints in the instruction (size, flavor, count, price, etc.).
- Prefer clicking the most relevant product or option.
- Avoid obvious loops or going back without reason.
- If nothing is clearly good, pick the safest progress-making action.

Return only JSON:
{"next_action": "<one of the candidate actions or 'none'>",
 "rationale": "<very short reason>"}"""

def _extract_json_block(txt: str) -> dict:
    candidates = []
    for m in re.finditer(r"\{.*?\}", txt, flags=re.DOTALL):
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            candidates.append(obj)
        except Exception:
            continue
    return candidates[-1] if candidates else {}

def make_webshop_traj_snippet(prompt_ctx: str, limit_chars: int = 1200) -> str:
    prompt_ctx = prompt_ctx or ""
    return prompt_ctx.strip() if len(prompt_ctx) <= limit_chars else prompt_ctx[-limit_chars:].strip()

def critic_llm_decide_action(
    instr_text: str,
    prompt_ctx: str,
    candidates: List[str],
    amb: Optional[dict] = None,
    to_print: bool = False,
) -> Tuple[Optional[str], int, int]:
    if not candidates:
        return None, 0, 0

    traj_snip = make_webshop_traj_snippet(prompt_ctx, limit_chars=1200)
    cand_block = "\n".join(f"- {a}" for a in candidates)

    amb_info = ""
    if amb is not None:
        amb_info = (
            f"\n[AMBIGUITY]\n"
            f"K={amb.get('K')}, Hbar={amb.get('Hbar')}, margin={amb.get('margin')}\n"
        )

    prompt = (
        f"{WEB_CRITIC_SYS_PROMPT}\n\n"
        f"INSTRUCTION:\n{(instr_text or '').strip()}\n\n"
        f"HISTORY:\n{traj_snip}\n"
        f"{amb_info}\n"
        f"CANDIDATE_ACTIONS:\n{cand_block}\n\n"
        f"JSON:"
    )

    text, prompt_tokens, gen_tokens = llm(prompt, stop=[], max_new_tokens=160)
    data = _extract_json_block(text)

    next_act = None
    if isinstance(data, dict):
        na = data.get("next_action")
        if isinstance(na, str):
            na = na.strip()
            if na.lower() == "none":
                next_act = None
            elif na in candidates:
                next_act = na
            else:
                cleaned = re.sub(r"^[Aa]ction\s*:\s*", "", na).strip()
                next_act = cleaned if cleaned in candidates else None

    if next_act is None and amb is not None:
        next_act = critic_suggest_action(amb)

    if to_print:
        print(f"[CRITIC-LLM] Chosen action: {next_act}")
        if isinstance(data, dict):
            rat = data.get("rationale", "")
            if rat:
                print(f"[CRITIC-LLM] Rationale: {rat}")

    return next_act, prompt_tokens, gen_tokens

# =========================================================
# WebShop env 및 유틸
# =========================================================
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (element.parent.name not in ignore and not isinstance(element, Comment))

def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
        url = f'{WEBSHOP_URL}/{session}'
    if page_type == 'search':
        url = f'{WEBSHOP_URL}/search_results/{session}/{query_string}/{page_num}'
    elif page_type == 'item':
        url = f'{WEBSHOP_URL}/item_page/{session}/{asin}/{query_string}/{page_num}/{options}'
    elif page_type == 'item_sub':
        url = f'{WEBSHOP_URL}/item_sub_page/{session}/{asin}/{query_string}/{page_num}/{subpage}/{options}'
    elif page_type == 'end':
        url = f'{WEBSHOP_URL}/done/{session}/{asin}/{options}'

    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))

    observation = ''
    option_type = ''
    options = {}
    asins = []
    cnt = 0
    prod_cnt = 0
    just_prod = 0
    for t in visible_texts:
        if t == '\n':
            continue
        if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '':
            continue
        if t.parent.name == 'button':
            processed_t = f'\n[{t}] '
        elif t.parent.name == 'label':
            if f"'{t}'" in url:
                processed_t = f'[[{t}]]'
            else:
                processed_t = f'[{t}]'
            options[str(t)] = option_type
        elif t.parent.get('class') == ["product-link"]:
            processed_t = f'\n[{t}] '
            if prod_cnt >= 3:
                processed_t = ''
            prod_cnt += 1
            asins.append(str(t))
            just_prod = 0
        else:
            processed_t = '\n' + str(t) + ' '
            if cnt < 2 and page_type != 'init':
                processed_t = ''
            if just_prod <= 2 and prod_cnt >= 4:
                processed_t = ''
            option_type = str(t)
            cnt += 1
        just_prod += 1
        observation += processed_t

    info = {}
    if options:
        info['option_types'] = options
    if asins:
        info['asins'] = asins

    if 'Your score (min 0.0, max 1.0)' in visible_texts:
        try:
            idx = visible_texts.index('Your score (min 0.0, max 1.0)')
            score_str = str(visible_texts[idx + 1]).strip()
            info['reward'] = float(score_str)
            observation = 'Your score (min 0.0, max 1.0): ' + score_str
        except Exception:
            info['reward'] = 0.0

    return clean_str(observation), info

class webshopEnv:
    def __init__(self):
        self.sessions = {}

    def step(self, session, action):
        done = False
        observation_ = None
        if action == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}
        elif action.startswith('think['):
            observation = 'OK.'
        elif action.startswith('search['):
            assert self.sessions[session]['page_type'] == 'init'
            query = action[7:-1]
            self.sessions[session] = {
                'session': session, 'page_type': 'search',
                'query_string': query, 'page_num': 1
            }
        elif action.startswith('click['):
            button = action[6:-1]
            if button == 'Buy Now':
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == 'Next >':
                assert False
            elif button == '< Prev':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                if self.sessions[session]['page_type'] == 'search':
                    assert False
                elif self.sessions[session]['page_type'] == 'item_sub':
                    self.sessions[session]['page_type'] = 'item'
                elif self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}
            elif button in ACTION_TO_TEMPLATE:
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if self.sessions[session]['page_type'] == 'search':
                    assert button in self.sessions[session].get('asins', [])
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif self.sessions[session]['page_type'] == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])
                    option_type = self.sessions[session]['option_types'][button]
                    if 'options' not in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False

        observation, info = webshop_text(**self.sessions[session])
        if observation_:
            observation = observation_
        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        return observation, reward, done

env = webshopEnv()

# =========================================================
# 인스트럭션 키워드 추출 + 옵션 휴리스틱
# =========================================================
def extract_instruction(observation: str) -> str:
    lines = observation.splitlines()
    inst_lines = []
    take = False
    for ln in lines:
        if "Instruction" in ln:
            take = True
            continue
        if take:
            if ln.strip() == "":
                break
            inst_lines.append(ln.strip())
    return " ".join(inst_lines).strip()

def extract_keywords_from_instruction(instr: str):
    instr = (instr or "").lower()
    tokens = re.findall(r"[a-z0-9\"\'\+\-]+", instr)
    stop = {
        "i", "need", "want", "looking", "for", "and", "the",
        "a", "an", "of", "with", "in", "it", "to", "me",
        "please", "find", "that"
    }
    return [t for t in tokens if t not in stop and len(t) > 1]

def heuristic_option_click(env, session_id: str, observation: str, instr_keywords):
    page_type = env.sessions.get(session_id, {}).get("page_type", None)
    if page_type != "item":
        return None

    buttons = extract_buttons_from_observation(observation)
    ignore = {
        "buy now", "back to search", "< prev",
        "description", "features", "reviews", "attributes"
    }
    cand_opts = [b for b in buttons if b.lower() not in ignore]

    if not cand_opts or not instr_keywords:
        return None

    def score_button(btn: str) -> int:
        bl = btn.lower()
        return sum(1 for kw in instr_keywords if kw in bl)

    scored = [(score_button(b), b) for b in cand_opts]
    scored.sort(reverse=True)
    best_score, best_btn = scored[0]
    if best_score >= 2:
        return f"click[{best_btn}]"
    return None

# =========================================================
# ICL 설정 및 유틸
# =========================================================
ICL_CFG = {
    "enable": True,
    "buffer_max": 16,
    "inject_k": 2,
    "max_chars_each": 2000
}
SUCCESS_ICL = deque(maxlen=ICL_CFG["buffer_max"])

def make_icl_example_from_prompt_ctx(prompt_ctx: str) -> str:
    body = (prompt_ctx or "").strip()
    if body.endswith("Action:"):
        body = body[:-len("Action:")].rstrip()
    if len(body) > ICL_CFG["max_chars_each"]:
        body = body[-ICL_CFG["max_chars_each"]:]
    return body + "\n"

def build_icl_block() -> str:
    if not ICL_CFG["enable"] or not SUCCESS_ICL:
        return ""
    k = min(ICL_CFG["inject_k"], len(SUCCESS_ICL))
    chosen = list(SUCCESS_ICL)[-k:]
    return "\n\n# Additional successful examples\n" + "\n\n".join(chosen) + "\n\n"

# =========================================================
# ✅ Actor LoRA SFT (성공 10개마다 micro-train)
# =========================================================
SAVE_DIR = Path("./trajectories")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SFT_DATA_PATH = SAVE_DIR / "actor_sft.jsonl"
SFT_DATA_PATH.touch(exist_ok=True)

SFT_CFG = {
    "enable": True,

    # 수집
    "buffer_max": 50000,
    "max_prompt_chars": 8000,   # 너무 긴 프롬프트는 뒷부분 유지
    "drop_think": False,        # think도 학습시키려면 False
    "min_prompt_chars": 50,

    # 트리거
    "train_every_success": 10,  # ✅ 성공 10개마다 학습
    "min_buffer_to_train": 200, # 너무 적으면 스킵 (에피소드당 step이 적어도 몇백은 금방 쌓임)

    # 학습 하이퍼
    "max_seq_len": 1024,
    "train_steps": 80,
    "train_batch_size": 2,
    "grad_accum": 8,
    "lr": 1e-5,

    # LoRA
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

SFT_BUFFER = deque(maxlen=SFT_CFG["buffer_max"])
SUCCESS_EPISODE_COUNT = 0
PEFT_READY = False

def _clip_tail(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[-n:]

def maybe_enable_lora():
    """
    model을 peft LoRA로 감싸기. 이미 감싸져 있으면 그대로.
    """
    global model, PEFT_READY
    if not SFT_CFG["enable"]:
        return

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        PEFT_READY = False
        return

    # 이미 peft면 skip
    if hasattr(model, "peft_config"):
        PEFT_READY = True
        return

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=SFT_CFG["lora_r"],
        lora_alpha=SFT_CFG["lora_alpha"],
        lora_dropout=SFT_CFG["lora_dropout"],
        target_modules=SFT_CFG["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    PEFT_READY = True

def append_sft_samples(samples: List[dict]):
    if not samples:
        return
    with open(SFT_DATA_PATH, "a", encoding="utf-8") as f:
        for ex in samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            SFT_BUFFER.append(ex)

def build_sft_sample(state_prompt: str, target_action: str, episode: str, step: int) -> Optional[dict]:
    """
    state_prompt는 보통 "... \n\nAction:" 으로 끝나는 상태.
    target_action은 "search[...]" 또는 "click[...]" 등.
    """
    sp = (state_prompt or "").strip()
    ta = (target_action or "").strip()
    if len(sp) < SFT_CFG["min_prompt_chars"] or not ta:
        return None
    if SFT_CFG["drop_think"] and ta.lower().startswith("think["):
        return None

    sp = _clip_tail(sp, SFT_CFG["max_prompt_chars"])

    return {
        "episode": episode,
        "step": step,
        "input": sp,
        "target": ta,
    }

def _batch_tokenize(ex_list: List[dict]) -> Dict[str, torch.Tensor]:
    """
    causal LM supervised fine-tuning:
    - input + " " + target 를 토크나이즈
    - labels에서 prompt 부분은 -100 마스킹
    """
    texts = []
    prompts = []
    for ex in ex_list:
        inp = ex["input"].rstrip()
        tgt = ex["target"].strip()
        prompts.append(inp)
        texts.append(inp + " " + tgt)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SFT_CFG["max_seq_len"],
    )

    labels = enc["input_ids"].clone()

    for i, inp in enumerate(prompts):
        inp_ids = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            max_length=SFT_CFG["max_seq_len"],
        )["input_ids"][0]
        L_inp = int(inp_ids.shape[0])
        labels[i, :L_inp] = -100

    enc["labels"] = labels
    return {k: v.to(DEVICE) for k, v in enc.items()}

def micro_sft_train():
    """
    성공 데이터 누적 후 짧게 학습(LoRA).
    """
    if not SFT_CFG["enable"]:
        return
    if not PEFT_READY:
        print("[SFT] peft not available or LoRA not enabled. (Data is still collected.)")
        return
    if len(SFT_BUFFER) < SFT_CFG["min_buffer_to_train"]:
        print(f"[SFT] Buffer too small ({len(SFT_BUFFER)}). Skip training.")
        return

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=SFT_CFG["lr"])

    steps = SFT_CFG["train_steps"]
    bs = SFT_CFG["train_batch_size"]
    ga = SFT_CFG["grad_accum"]

    print(f"[SFT] Micro-training start: steps={steps}, bs={bs}, grad_accum={ga}, buffer={len(SFT_BUFFER)}")

    optim.zero_grad(set_to_none=True)
    data = list(SFT_BUFFER)

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
            print(f"[SFT] step={step}/{steps} | loss={float(loss.item())*ga:.4f}")

    model.eval()
    print("[SFT] Micro-training done.")

# =========================================================
# 샘플 프롬프트들 (few-shot)
# =========================================================
prompt1 = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""

BASE_PROMPT_FEWSHOT = prompt1

# =========================================================
# 에피소드 실행 (+ ICL + SFT 샘플 수집)
# =========================================================
def webshop_run(idx, prompt, to_print=True):
    """
    return:
      reward, total_in, total_out, traj_len, amb_logs, icl_example,
      success_sft_samples(list), steps_used  🔴 steps_used 추가
    """
    action = 'reset'
    init_prompt = prompt
    prompt_ctx = ''
    total_in, total_out = 0, 0

    step_amb_logs = []
    last_valid_observation = None
    instr_text = None
    instr_keywords = []

    # ✅ 성공 시에만 반환할 SFT 샘플들(한 에피소드 분)
    episode_sft_samples = []

    steps_used = 0  # 🔴 step 카운터

    for step_i in range(15):
        steps_used = step_i + 1  # 현재까지 사용한 step 수

        try:
            observation, reward, done = env.step(idx, action)
        except AssertionError:
            observation, reward, done = 'Invalid action!', 0.0, False

        if action.startswith('think'):
            observation = 'OK.'

        if step_i == 0:
            instr_text = extract_instruction(observation)
            instr_keywords = extract_keywords_from_instruction(instr_text or "")

        if observation not in ('OK.', 'Invalid action!'):
            last_valid_observation = observation

        if to_print:
            print(f'Action: {action}\nObservation: {observation}\n')
            sys.stdout.flush()

        # prompt 업데이트
        if step_i:
            prompt_ctx += f' {action}\nObservation: {observation}\n\nAction:'
        else:
            prompt_ctx += f'{observation}\n\nAction:'

        # 종료면 반환
        if done:
            traj_len = len(init_prompt + prompt_ctx)
            icl_example = None
            if reward >= 1.0 and ICL_CFG["enable"]:
                icl_example = make_icl_example_from_prompt_ctx(prompt_ctx)
            # 🔴 steps_used 함께 반환
            return (
                reward,
                total_in,
                total_out,
                traj_len,
                step_amb_logs,
                icl_example,
                episode_sft_samples,
                steps_used,
            )

        # ✅ 1) ITEM에서 옵션 휴리스틱
        heur_action = heuristic_option_click(env, idx, observation, instr_keywords)
        if heur_action is not None:
            if to_print:
                print(f"[HEURISTIC] Force option click -> {heur_action}")

            # ✅ SFT 샘플 수집: 이 step에서 모델이 보게 될 state_prompt
            state_prompt = (init_prompt + prompt_ctx).strip()
            sample = build_sft_sample(state_prompt, heur_action, episode=str(idx), step=step_i)
            if sample is not None:
                episode_sft_samples.append(sample)

            action = heur_action
            continue

        # SAG candidate용 obs
        obs_for_candidates = observation
        if observation in ('OK.', 'Invalid action!') and last_valid_observation is not None:
            obs_for_candidates = last_valid_observation

        cands = admissible_click_actions(env, idx, obs_for_candidates)

        amb = ambiguity_signals_from_candidates(init_prompt + prompt_ctx, cands) if len(cands) >= 2 else None
        if amb is not None:
            step_amb_logs.append(amb)
            if to_print:
                print(f"[AMB] K={amb['K']} Hbar={amb['Hbar']:.3f} margin={amb['margin']:.3f}")
                for a, p in amb["topk"][:3]:
                    print(f"      {a}  q={p:.3f}")
                print()

        # ✅ 2) critic
        used_critic = False
        if should_call_critic(amb):
            crit_act, cin_tok, cout_tok = critic_llm_decide_action(
                instr_text=instr_text,
                prompt_ctx=prompt_ctx,
                candidates=cands,
                amb=amb,
                to_print=to_print,
            )
            total_in += cin_tok
            total_out += cout_tok

            if crit_act is not None:
                used_critic = True
                if to_print:
                    print(
                        f"[CRITIC] Triggered at step {step_i}, episode {idx}. "
                        f"(Hbar={amb['Hbar']:.3f}, margin={amb['margin']:.3f})"
                    )
                    print(f"[CRITIC] LLM suggested action: {crit_act}")

                # ✅ SFT 샘플 수집
                state_prompt = (init_prompt + prompt_ctx).strip()
                sample = build_sft_sample(state_prompt, crit_act, episode=str(idx), step=step_i)
                if sample is not None:
                    episode_sft_samples.append(sample)

                action = crit_act

        # ✅ 3) actor LLM
        if not used_critic:
            full_ctx = init_prompt + prompt_ctx
            max_len = 6400
            ctx = full_ctx[-max_len:] if len(full_ctx) > max_len else full_ctx

            text, in_tok, out_tok = llm(ctx, stop=['\n'], max_new_tokens=100)
            total_in += in_tok
            total_out += out_tok
            action = text.lstrip(' ')

            # ✅ SFT 샘플 수집: actor가 고른 action을 target으로
            state_prompt = (init_prompt + prompt_ctx).strip()
            sample = build_sft_sample(state_prompt, action, episode=str(idx), step=step_i)
            if sample is not None:
                episode_sft_samples.append(sample)

        # ✅ 4) grammar guard
        page_type = env.sessions.get(idx, {}).get("page_type", None)
        if action.startswith('search[') and page_type == 'item':
            if to_print:
                print("[GRAMMAR] search[...] not allowed on page_type=item. Force action -> click[Back to Search]")
            action = 'click[Back to Search]'

    # max step까지 가도 done 안되면 실패 종료
    traj_len = len(init_prompt + prompt_ctx)
    steps_used = 15
    return 0.0, total_in, total_out, traj_len, step_amb_logs, None, [], steps_used

# =========================================================
# 전체 에피소드 반복 + 통계/저장 + ICL + (성공10개마다 LoRA SFT)
# =========================================================
def run_episodes(base_prompt, n=50, seed: int = 42):
    global SUCCESS_EPISODE_COUNT
    set_global_seed(seed)

    # ✅ LoRA 준비 (peft 있으면 model을 감쌈)
    maybe_enable_lora()
    if SFT_CFG["enable"] and not PEFT_READY:
        print("[SFT] peft not found -> actor LoRA training disabled (data collection only).")

    rs = []
    cnt = 0
    total_in_tok, total_out_tok = 0, 0
    input_outputs = []
    run_start_time = time.time()

    success_steps = []  # 🔴 성공 에피소드의 step 수 저장

    for i in range(n):
        print('-----------------')
        print(i)

        icl_block = build_icl_block()
        prompt = base_prompt + icl_block

        start_ep = time.time()
        try:
            # 🔴 steps_used 함께 받기
            r, in_tok, out_tok, traj_len, amb_logs, icl_example, sft_samples, steps_used = webshop_run(
                f'fixed_{i}', prompt, to_print=True
            )
            total_in_tok += in_tok
            total_out_tok += out_tok

            # ✅ 성공이면 ICL 업데이트 + SFT 샘플 적재 + (성공10개마다) micro-train
            if r >= 1.0:
                success_steps.append(steps_used)  # 🔴 성공 step 기록

                if icl_example and ICL_CFG["enable"]:
                    SUCCESS_ICL.append(icl_example)
                    print(f"[ICL] Added success example. Buffer size = {len(SUCCESS_ICL)}")

                if SFT_CFG["enable"] and sft_samples:
                    SUCCESS_EPISODE_COUNT += 1
                    append_sft_samples(sft_samples)
                    print(f"[SFT] Collected: +{len(sft_samples)} | buffer={len(SFT_BUFFER)} | success_eps={SUCCESS_EPISODE_COUNT}")

                    if (SUCCESS_EPISODE_COUNT % SFT_CFG["train_every_success"] == 0):
                        micro_sft_train()

            end_ep = time.time()
            ep_time = end_ep - start_ep

            input_outputs.append({
                "episode_idx": i,
                "reward": float(r),
                "success": 1 if r >= 1.0 else 0,
                "prompt_tokens": int(in_tok),
                "gen_tokens": int(out_tok),
                "total_tokens": int(in_tok + out_tok),
                "traj_length_chars": int(traj_len),
                "time_sec": ep_time,
                "ambiguity_logs": amb_logs,
                "steps_used": int(steps_used),   # 🔴 per-ep step 수 저장
            })
        except AssertionError:
            r, in_tok, out_tok, traj_len, amb_logs = 0.0, 0, 0, 0, []
            cnt += 1
            end_ep = time.time()
            ep_time = end_ep - start_ep

            input_outputs.append({
                "episode_idx": i,
                "reward": 0.0,
                "success": 0,
                "prompt_tokens": 0,
                "gen_tokens": 0,
                "total_tokens": 0,
                "traj_length_chars": 0,
                "time_sec": ep_time,
                "ambiguity_logs": amb_logs,
                "steps_used": 0,
            })

        print(f"Episode {i} took {ep_time:.2f} seconds")

        rs.append(r)
        r_avg = sum(rs) / len(rs)
        sr = len([_ for _ in rs if _ == 1]) / len(rs)
        fr = cnt / len(rs)
        print(f"{i+1} episodes | avg reward={r_avg:.3f}, success={sr:.3f}, fail={fr:.3f}")
        print('-------------')

    end_all = time.time()
    total_time = end_all - run_start_time
    total_cnt = len(input_outputs)
    total_succ = len([ep for ep in input_outputs if ep["success"] == 1])
    overall_success = total_succ / total_cnt if total_cnt > 0 else 0.0

    avg_traj_len = (sum(io["traj_length_chars"] for io in input_outputs) / total_cnt) if total_cnt > 0 else 0.0
    avg_time_ep = (sum(io["time_sec"] for io in input_outputs) / total_cnt) if total_cnt > 0 else 0.0

    total_prompt_tokens = total_in_tok
    total_gen_tokens = total_out_tok
    total_tokens = total_prompt_tokens + total_gen_tokens
    avg_prompt_tokens_ep = total_prompt_tokens / total_cnt if total_cnt > 0 else 0.0
    avg_gen_tokens_ep = total_gen_tokens / total_cnt if total_cnt > 0 else 0.0
    avg_total_tokens_ep = total_tokens / total_cnt if total_cnt > 0 else 0.0

    # 🔴 성공 에피소드 평균 step 수
    avg_success_steps = (sum(success_steps) / len(success_steps)) if success_steps else 0.0

    summary = {
        "dataset": "WebShop",
        "split": "test",
        "model_name_or_path": MODEL_NAME,
        "actor_lora_enabled": bool(PEFT_READY),
        "sft_data_path": str(SFT_DATA_PATH),
        "sft_buffer_size": int(len(SFT_BUFFER)),
        "success_eps_for_sft": int(SUCCESS_EPISODE_COUNT),

        "total_episodes": int(total_cnt),
        "total_success": float(total_succ),
        "overall_success_rate": overall_success,

        "total_prompt_tokens": int(total_prompt_tokens),
        "total_gen_tokens": int(total_gen_tokens),
        "total_tokens": int(total_tokens),
        "avg_prompt_tokens_per_episode": avg_prompt_tokens_ep,
        "avg_gen_tokens_per_episode": avg_gen_tokens_ep,
        "avg_total_tokens_per_episode": avg_total_tokens_ep,
        "avg_traj_length_chars_per_episode": avg_traj_len,
        "total_time_sec": total_time,
        "avg_time_per_episode_sec": avg_time_ep,

        # 🔴 성공 에피소드 step 통계
        "avg_success_steps": avg_success_steps,
        "success_steps": [int(s) for s in success_steps],

        "episodes": input_outputs,
    }

    print("\n=== Run Summary ===")
    print(f"Total {n} episodes took {total_time:.2f} sec (avg {total_time/n:.2f} sec/ep)")
    print(f"Total Input Tokens  : {total_prompt_tokens}")
    print(f"Total Output Tokens : {total_gen_tokens}")
    print(f"Total Tokens        : {total_tokens}")
    print(f"SFT buffer          : {len(SFT_BUFFER)} | success_eps_for_sft={SUCCESS_EPISODE_COUNT}")
    print(f"Actor LoRA enabled  : {PEFT_READY}")
    print(f"Avg success steps   : {avg_success_steps:.2f}")  # 🔴 콘솔에도 출력

    os.makedirs("result", exist_ok=True)
    model_tag = MODEL_NAME.split('/')[-1] if isinstance(MODEL_NAME, str) else "unknown_model"
    save_path = os.path.join("result", f"webshop_{model_tag}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"결과 저장 완료: {save_path}")
    print(f"SFT 데이터 저장: {SFT_DATA_PATH}")

    return rs

# 실행 예시
if __name__ == "__main__":
    res1 = run_episodes(BASE_PROMPT_FEWSHOT, n=500, seed=42)
