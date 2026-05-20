from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prism")


@dataclass
class PRISMConfig:
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    encoder_model_name: str = "microsoft/deberta-v3-base"
    benchmarks: Tuple[str, ...] = ("alfworld", "webshop", "scienceworld")
    k_collect: int = 4
    k_collect_finetuned: int = 2
    decode_temperature: float = 0.7
    decode_top_p: float = 0.95
    max_new_tokens: int = 256
    m_continuations: int = 8
    n_step_labels: int = 60000
    lambda_bce: float = 1.0
    lambda_mse: float = 0.5
    prm_epochs: int = 20
    prm_lr: float = 1.5e-5
    prm_batch_size: int = 32
    prm_weight_decay: float = 0.01
    prm_warmup_ratio: float = 0.10
    grad_clip: float = 1.0
    conf_epochs: int = 20
    conf_lr: float = 1.0e-5
    conf_batch_size: int = 32
    conf_warmup_ratio: float = 0.10
    platt_validation_size: int = 670
    iql_expectile_tau: float = 0.7
    iql_gamma: float = 0.99
    iql_polyak: float = 0.005
    iql_lr: float = 5.0e-5
    iql_batch_size: int = 64
    iql_steps: int = 50000
    iql_warmup_ratio: float = 0.10
    alpha_reward_mix: float = 0.7
    awr_beta: float = 3.0
    awr_delta_max: float = 4.0
    awr_negative_advantage_weight: float = 0.1
    awr_epochs: int = 1
    awr_lr: float = 1.0e-4
    awr_batch_size: int = 16
    awr_warmup_ratio: float = 0.05
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    k_candidates: int = 3
    tau_low: float = 0.45
    tau_abort: float = 0.20
    r_max: int = 2
    horizon_alfworld: int = 30
    horizon_webshop: int = 25
    horizon_scienceworld: int = 30
    seed: int = 42
    seeds_eval: Tuple[int, ...] = (1, 7, 13, 23, 42)
    output_dir: str = "./prism_outputs"
    log_every: int = 100
    save_every: int = 5000
    bf16: bool = True
    gradient_checkpointing: bool = True
    webshop_success_threshold: float = 0.9
    scienceworld_success_threshold: float = 90.0
    encoder_max_length: int = 512
    policy_max_length: int = 4096
    action_max_tokens: int = 256

    def horizon(self, benchmark: str) -> int:
        return {
            "alfworld": self.horizon_alfworld,
            "webshop": self.horizon_webshop,
            "scienceworld": self.horizon_scienceworld,
        }[benchmark]

    def success_threshold(self, benchmark: str) -> float:
        return {
            "alfworld": 1.0,
            "webshop": self.webshop_success_threshold,
            "scienceworld": self.scienceworld_success_threshold,
        }[benchmark]


@dataclass
class StepTransition:
    benchmark: str
    step_index: int
    trajectory_id: str
    env_reward: float
    terminal: bool


@dataclass
class Trajectory:
    trajectory_id: str
    benchmark: str
    task_id: str
    goal_text: str
    states: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    success: bool = False
    final_reward: float = 0.0
    length: int = 0

    def transitions(self) -> List[StepTransition]:
        result = []
        for t in range(self.length):
            terminal = (t == self.length - 1)
            result.append(StepTransition(
                benchmark=self.benchmark,
                step_index=t,
                trajectory_id=self.trajectory_id,
                env_reward=self.rewards[t],
                terminal=terminal,
            ))
        return result

    def prefix_text(self, t: int) -> str:
        parts = [f"Goal: {self.goal_text}"]
        for i in range(min(t, self.length)):
            parts.append(f"Obs {i}: {self.states[i]}")
            parts.append(f"Act {i}: {self.actions[i]}")
        if t < len(self.states):
            parts.append(f"Obs {t}: {self.states[t]}")
        return "\n".join(parts)

    def prefix_action_text(self, t: int) -> str:
        if t >= self.length:
            return self.prefix_text(t)
        return self.prefix_text(t) + "\n[ACT] " + self.actions[t]


REACT_SYSTEM_PROMPT = (
    "You are an interactive agent operating in a textual environment. "
    "At each step you produce a short rationale prefixed with 'Thought:' "
    "and exactly one action prefixed with 'Action:'. Actions must conform "
    "to the environment-specific tool grammar."
)

REACT_EXEMPLAR_ALFWORLD = (
    "Goal: put a clean apple on the table.\n"
    "Obs 0: You are in the middle of the room.\n"
    "Thought: I should find an apple first.\n"
    "Action: go to fridge 1\n"
    "Obs 1: The fridge 1 is closed.\n"
    "Thought: I will open the fridge.\n"
    "Action: open fridge 1\n"
)

REACT_EXEMPLAR_WEBSHOP = (
    "Goal: buy a red cotton t-shirt under $25.\n"
    "Obs 0: WebShop home page.\n"
    "Thought: I should search for a red cotton t-shirt.\n"
    "Action: search[red cotton t-shirt]\n"
)

REACT_EXEMPLAR_SCIENCEWORLD = (
    "Goal: measure the melting point of ice.\n"
    "Obs 0: You are in the kitchen.\n"
    "Thought: I need a thermometer and a stove.\n"
    "Action: open cupboard\n"
)

EXEMPLARS = {
    "alfworld": REACT_EXEMPLAR_ALFWORLD,
    "webshop": REACT_EXEMPLAR_WEBSHOP,
    "scienceworld": REACT_EXEMPLAR_SCIENCEWORLD,
}


class BaseEnv:
    def __init__(self, benchmark: str, config: PRISMConfig):
        self.benchmark = benchmark
        self.config = config

    def list_train_tasks(self) -> List[str]:
        raise NotImplementedError

    def list_eval_tasks(self) -> List[str]:
        raise NotImplementedError

    def reset(self, task_id: str) -> Tuple[str, str]:
        raise NotImplementedError

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        raise NotImplementedError


class AlfWorldEnv(BaseEnv):
    def __init__(self, benchmark: str, config: PRISMConfig):
        super().__init__(benchmark, config)
        import alfworld.agents.environment as alf_env
        import alfworld.agents.modules.generic as generic
        self._cfg = generic.load_config()
        self._train_env = alf_env.AlfredTWEnv(self._cfg, train_eval="train")
        self._train_env = self._train_env.init_env(batch_size=1)
        self._eval_env = alf_env.AlfredTWEnv(self._cfg, train_eval="eval_out_of_distribution")
        self._eval_env = self._eval_env.init_env(batch_size=1)
        self._active = None
        self._train_ids = self._collect_gamefiles(self._train_env)
        self._eval_ids = self._collect_gamefiles(self._eval_env)

    def _collect_gamefiles(self, env) -> List[str]:
        try:
            return [str(gf) for gf in getattr(env, "gamefiles", [])]
        except Exception:
            return []

    def list_train_tasks(self) -> List[str]:
        if self._train_ids:
            return list(self._train_ids)
        return [f"alfworld_train_{i}" for i in range(3553)]

    def list_eval_tasks(self) -> List[str]:
        if self._eval_ids:
            return list(self._eval_ids)
        return [f"alfworld_eval_{i}" for i in range(134)]

    def reset(self, task_id: str) -> Tuple[str, str]:
        self._active = self._eval_env if task_id in self._eval_ids else self._train_env
        obs, info = self._active.reset()
        obs_text = obs[0] if isinstance(obs, list) else str(obs)
        goal = str(info.get("extra.gamefile", task_id)) if isinstance(info, dict) else str(task_id)
        return obs_text, goal

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self._active.step([action])
        next_obs = obs[0] if isinstance(obs, list) else str(obs)
        reward = float(scores[0]) if hasattr(scores, "__len__") else float(scores)
        done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
        info = infos[0] if isinstance(infos, list) and infos else (infos if isinstance(infos, dict) else {})
        info["success"] = bool(info.get("won", reward >= 1.0))
        return next_obs, reward, done, info


class ScienceWorldEnv(BaseEnv):
    def __init__(self, benchmark: str, config: PRISMConfig):
        super().__init__(benchmark, config)
        from scienceworld import ScienceWorldEnv as SWEnv
        self._env = SWEnv("", "", envStepLimit=config.horizon_scienceworld)
        self._task_names = self._env.getTaskNames()
        self._train_tasks = [f"{name}::var{v}" for name in self._task_names for v in range(5)]
        self._eval_tasks = [f"{name}::var{v}" for name in self._task_names for v in range(5, 6)]
        self._active_task: Optional[str] = None
        self._reward_acc = 0.0

    def list_train_tasks(self) -> List[str]:
        return list(self._train_tasks)[:1371]

    def list_eval_tasks(self) -> List[str]:
        return list(self._eval_tasks)[:270]

    def reset(self, task_id: str) -> Tuple[str, str]:
        name, var_part = task_id.split("::var")
        variation = int(var_part)
        self._env.load(name, variation, "easy", generateGoldPath=False)
        obs, info = self._env.reset()
        self._active_task = task_id
        self._reward_acc = 0.0
        goal = self._env.getTaskDescription()
        return str(obs), str(goal)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, is_completed, info = self._env.step(action)
        self._reward_acc = max(self._reward_acc, float(reward))
        success = bool(is_completed) and self._reward_acc >= self.config.scienceworld_success_threshold
        return str(obs), float(reward), bool(is_completed), {"success": success, "raw_reward": float(reward)}


class WebShopEnv(BaseEnv):
    def __init__(self, benchmark: str, config: PRISMConfig):
        super().__init__(benchmark, config)
        try:
            from web_agent_site.envs import WebAgentTextEnv
        except ImportError as exc:
            raise ImportError(
                "WebShop is not installed. Clone https://github.com/princeton-nlp/WebShop and run setup.sh, "
                "then add the WebShop repo root to PYTHONPATH."
            ) from exc
        self._env = WebAgentTextEnv(observation_mode="text", num_products=None)
        self._active_task: Optional[str] = None

    def list_train_tasks(self) -> List[str]:
        return [f"webshop_train_{i}" for i in range(1000)]

    def list_eval_tasks(self) -> List[str]:
        return [f"webshop_eval_{i}" for i in range(500)]

    def reset(self, task_id: str) -> Tuple[str, str]:
        session = int(task_id.rsplit("_", 1)[-1])
        obs, info = self._env.reset(session=session)
        self._active_task = task_id
        goal = info.get("instruction_text", str(task_id)) if isinstance(info, dict) else str(task_id)
        return str(obs), str(goal)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        success = bool(done) and float(reward) >= self.config.webshop_success_threshold
        return str(obs), float(reward), bool(done), {"success": success, "raw_reward": float(reward)}


def build_env(benchmark: str, config: PRISMConfig) -> BaseEnv:
    if benchmark == "alfworld":
        return AlfWorldEnv(benchmark, config)
    if benchmark == "scienceworld":
        return ScienceWorldEnv(benchmark, config)
    if benchmark == "webshop":
        return WebShopEnv(benchmark, config)
    raise ValueError(f"Unknown benchmark {benchmark}")


class BasePolicy:
    def __init__(self, config: PRISMConfig, lora_adapter_path: Optional[str] = None):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = torch.bfloat16 if (config.bf16 and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if lora_adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        self.lora_attached = lora_adapter_path is not None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def attach_lora(self):
        if self.lora_attached:
            return
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=list(self.config.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.lora_attached = True

    def build_prompt(self, benchmark: str, prefix_text: str, reflection: Optional[str] = None) -> str:
        exemplar = EXEMPLARS[benchmark]
        parts = [REACT_SYSTEM_PROMPT, "", exemplar, "", prefix_text]
        if reflection is not None:
            parts.append("")
            parts.append(f"Reflection: {reflection}")
        parts.append("Thought:")
        return "\n".join(parts)

    @torch.no_grad()
    def sample(self, prompt: str, k: int = 1, temperature: float = 0.7,
               max_new_tokens: Optional[int] = None) -> List[str]:
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=self.config.policy_max_length).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=self.config.decode_top_p,
            num_return_sequences=k,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        outputs = []
        for seq in out:
            text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            outputs.append(text)
        return outputs

    def parse_action(self, generated_text: str) -> str:
        if "Action:" in generated_text:
            tail = generated_text.split("Action:", 1)[1]
            line = tail.split("\n")[0].strip()
            return line
        return generated_text.split("\n")[0].strip()


def rollout_one_trajectory(policy: BasePolicy, env: BaseEnv, task_id: str,
                           horizon: int, config: PRISMConfig,
                           reflection: Optional[str] = None) -> Optional[Trajectory]:
    obs, goal = env.reset(task_id)
    traj = Trajectory(
        trajectory_id=f"{task_id}_{int(time.time()*1000)}_{random.randint(0,99999)}",
        benchmark=env.benchmark,
        task_id=task_id,
        goal_text=goal,
    )
    traj.states.append(obs)
    for t in range(horizon):
        prefix_text = traj.prefix_text(t)
        prompt = policy.build_prompt(env.benchmark, prefix_text, reflection=reflection if t == 0 else None)
        samples = policy.sample(prompt, k=1, temperature=config.decode_temperature)
        action = policy.parse_action(samples[0])
        traj.actions.append(action)
        next_obs, reward, terminal, info = env.step(action)
        traj.rewards.append(reward)
        traj.states.append(next_obs)
        traj.length += 1
        if terminal:
            traj.success = bool(info.get("success", reward >= env.config.success_threshold(env.benchmark)))
            traj.final_reward = reward
            break
    if traj.length == 0:
        return None
    return traj


def collect_trajectory_buffer(policies: List[Tuple[BasePolicy, int]], envs: Dict[str, BaseEnv],
                              config: PRISMConfig,
                              max_tasks_per_env: Optional[int] = None) -> List[Trajectory]:
    buffer: List[Trajectory] = []
    for benchmark, env in envs.items():
        tasks = env.list_train_tasks()
        if max_tasks_per_env is not None:
            tasks = tasks[:max_tasks_per_env]
        horizon = config.horizon(benchmark)
        for task_id in tasks:
            for policy, n_rollouts in policies:
                for _ in range(n_rollouts):
                    tr = rollout_one_trajectory(policy, env, task_id, horizon, config)
                    if tr is not None and tr.length >= 3:
                        buffer.append(tr)
    seen = set()
    deduped: List[Trajectory] = []
    for tr in buffer:
        key = (tr.task_id, tuple(tr.actions))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tr)
    return deduped


def continue_rollout_from_prefix(parent: Trajectory, t: int, policy: BasePolicy,
                                 env: BaseEnv, config: PRISMConfig) -> bool:
    horizon = config.horizon(parent.benchmark)
    if t + 1 >= len(parent.states):
        return parent.success
    init_states = list(parent.states[:t + 2])
    init_actions = list(parent.actions[:t + 1])
    init_rewards = list(parent.rewards[:t + 1])
    cur = Trajectory(
        trajectory_id=f"{parent.trajectory_id}_cont_{random.randint(0,9999999)}",
        benchmark=parent.benchmark,
        task_id=parent.task_id,
        goal_text=parent.goal_text,
        states=init_states,
        actions=init_actions,
        rewards=init_rewards,
        length=t + 1,
    )
    env.reset(parent.task_id)
    for past_action in init_actions:
        _, _, terminal_replay, _ = env.step(past_action)
        if terminal_replay:
            return cur.actions[-1] == parent.actions[-1] and parent.success
    remaining = horizon - cur.length
    if remaining <= 0:
        return parent.success
    for _ in range(remaining):
        prefix_text = cur.prefix_text(cur.length)
        prompt = policy.build_prompt(cur.benchmark, prefix_text)
        samples = policy.sample(prompt, k=1, temperature=config.decode_temperature)
        action = policy.parse_action(samples[0])
        cur.actions.append(action)
        next_obs, reward, terminal, info = env.step(action)
        cur.states.append(next_obs)
        cur.rewards.append(reward)
        cur.length += 1
        if terminal:
            return bool(info.get("success", reward >= env.config.success_threshold(cur.benchmark)))
    return False


def monte_carlo_step_labels(buffer: List[Trajectory], policy: BasePolicy,
                            envs: Dict[str, BaseEnv], config: PRISMConfig) -> List[Dict[str, Any]]:
    triples: List[Dict[str, Any]] = []
    prefix_pool: List[Tuple[Trajectory, int]] = []
    for tr in buffer:
        for t in range(tr.length):
            prefix_pool.append((tr, t))
    succ = [p for p in prefix_pool if p[0].success]
    fail = [p for p in prefix_pool if not p[0].success]
    n_each = min(config.n_step_labels // 2, len(succ), len(fail))
    if n_each == 0:
        sampled = random.sample(prefix_pool, min(config.n_step_labels, len(prefix_pool)))
    else:
        sampled = random.sample(succ, n_each) + random.sample(fail, n_each)
    random.shuffle(sampled)
    for tr, t in sampled:
        env = envs[tr.benchmark]
        successes = 0
        for _ in range(config.m_continuations):
            ok = continue_rollout_from_prefix(tr, t, policy, env, config)
            if ok:
                successes += 1
        q_hat = successes / float(config.m_continuations)
        y_label = 1 if q_hat > 0.5 else 0
        triples.append({
            "prefix_text": tr.prefix_text(t),
            "action_text": tr.actions[t] if t < len(tr.actions) else "noop",
            "q_hat": q_hat,
            "y_label": y_label,
            "trajectory_id": tr.trajectory_id,
            "step_index": t,
            "benchmark": tr.benchmark,
            "outcome": int(tr.success),
        })
    return triples


class EncoderBackbone(nn.Module):
    def __init__(self, model_name: str, max_length: int = 512):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.max_length = max_length

    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(device)
        out = self.encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts


class ProcessRewardModel(nn.Module):
    def __init__(self, backbone: EncoderBackbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden_size, 1)

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        h = self.backbone.encode(texts, device)
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def score(self, texts: List[str], device: torch.device) -> torch.Tensor:
        logits = self.forward(texts, device)
        return torch.sigmoid(logits)


class ConfidenceHead(nn.Module):
    def __init__(self, backbone: EncoderBackbone, mlp_hidden: int = 256):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(backbone.hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 1)
        self.platt_a: Optional[float] = None
        self.platt_b: Optional[float] = None

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        h = self.backbone.encode(texts, device)
        h = torch.tanh(self.fc1(h))
        return self.fc2(h).squeeze(-1)

    @torch.no_grad()
    def confidence_raw(self, texts: List[str], device: torch.device) -> torch.Tensor:
        return torch.sigmoid(self.forward(texts, device))

    @torch.no_grad()
    def confidence_calibrated(self, texts: List[str], device: torch.device) -> torch.Tensor:
        raw = self.confidence_raw(texts, device).clamp(1e-6, 1 - 1e-6)
        if self.platt_a is None or self.platt_b is None:
            return raw
        logit = torch.log(raw / (1 - raw))
        adjusted = (logit - self.platt_a) / max(self.platt_b, 1e-6)
        return torch.sigmoid(adjusted)


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings).squeeze(-1)


class QValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings).squeeze(-1)


class PRMDataset(Dataset):
    def __init__(self, triples: List[Dict[str, Any]]):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        t = self.triples[idx]
        return {
            "text": t["prefix_text"] + "\n[ACT] " + t["action_text"],
            "q_hat": float(t["q_hat"]),
            "y_label": int(t["y_label"]),
        }


class ConfidenceDataset(Dataset):
    def __init__(self, buffer: List[Trajectory], samples_per_traj: int = 3):
        self.items: List[Tuple[str, int]] = []
        for tr in buffer:
            for _ in range(samples_per_traj):
                if tr.length < 1:
                    continue
                t = random.randint(0, max(tr.length - 1, 0))
                self.items.append((tr.prefix_text(t), int(tr.success)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, y = self.items[idx]
        return {"text": text, "y": int(y)}


def collate_prm(batch):
    return {
        "texts": [b["text"] for b in batch],
        "q_hat": torch.tensor([b["q_hat"] for b in batch], dtype=torch.float32),
        "y_label": torch.tensor([b["y_label"] for b in batch], dtype=torch.float32),
    }


def collate_conf(batch):
    return {
        "texts": [b["text"] for b in batch],
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
    }


def train_prm(prm: ProcessRewardModel, triples: List[Dict[str, Any]], config: PRISMConfig, device: torch.device):
    train_n = int(0.9 * len(triples))
    train_triples = triples[:train_n]
    val_triples = triples[train_n:]
    train_ds = PRMDataset(train_triples)
    val_ds = PRMDataset(val_triples)
    train_loader = DataLoader(train_ds, batch_size=config.prm_batch_size, shuffle=True, collate_fn=collate_prm)
    val_loader = DataLoader(val_ds, batch_size=config.prm_batch_size, shuffle=False, collate_fn=collate_prm)
    optimizer = torch.optim.AdamW(prm.parameters(), lr=config.prm_lr, weight_decay=config.prm_weight_decay)
    total_steps = max(1, config.prm_epochs * len(train_loader))
    warmup_steps = int(config.prm_warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    prm.to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()
    step = 0
    for epoch in range(config.prm_epochs):
        prm.train()
        for batch in train_loader:
            logits = prm.forward(batch["texts"], device)
            probs = torch.sigmoid(logits)
            y = batch["y_label"].to(device)
            q = batch["q_hat"].to(device)
            loss = config.lambda_bce * bce_fn(logits, y) + config.lambda_mse * mse_fn(probs, q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prm.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                logger.info(f"[PRM] epoch={epoch} step={step} loss={loss.item():.4f}")
        prm.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in val_loader:
                logits = prm.forward(batch["texts"], device)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == batch["y_label"].to(device)).sum().item()
                total += preds.shape[0]
            acc = correct / max(total, 1)
            logger.info(f"[PRM] epoch={epoch} val_step_label_acc={acc:.4f}")
    return prm


def expected_calibration_error_and_brier(logits: List[float], labels: List[int],
                                         platt_a: Optional[float], platt_b: Optional[float],
                                         n_bins: int = 10) -> Tuple[float, float]:
    raw = 1.0 / (1.0 + np.exp(-np.array(logits)))
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    if platt_a is not None and platt_b is not None:
        raw_logits = np.log(raw / (1 - raw))
        adj = (raw_logits - platt_a) / max(platt_b, 1e-6)
        probs = 1.0 / (1.0 + np.exp(-adj))
    else:
        probs = raw
    labels_arr = np.array(labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for m in range(n_bins):
        lo, hi = bin_edges[m], bin_edges[m + 1]
        if m < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = labels_arr[mask].mean()
        conf_bin = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf_bin)
    brier = float(((probs - labels_arr) ** 2).mean())
    return float(ece), brier


def train_confidence_head(conf: ConfidenceHead, buffer: List[Trajectory],
                          config: PRISMConfig, device: torch.device):
    ds = ConfidenceDataset(buffer)
    n = len(ds)
    n_train = max(1, n - config.platt_validation_size)
    train_ds = torch.utils.data.Subset(ds, range(n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n))
    train_loader = DataLoader(train_ds, batch_size=config.conf_batch_size, shuffle=True, collate_fn=collate_conf)
    optimizer = torch.optim.AdamW(conf.parameters(), lr=config.conf_lr, weight_decay=0.01)
    total_steps = max(1, config.conf_epochs * len(train_loader))
    warmup_steps = int(config.conf_warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    conf.to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    step = 0
    for epoch in range(config.conf_epochs):
        conf.train()
        for batch in train_loader:
            logits = conf.forward(batch["texts"], device)
            y = batch["y"].to(device)
            loss = bce_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conf.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                logger.info(f"[CONF] epoch={epoch} step={step} loss={loss.item():.4f}")
    conf.eval()
    val_logits: List[float] = []
    val_labels: List[int] = []
    with torch.no_grad():
        for i in range(0, len(val_ds), config.conf_batch_size):
            items = [val_ds[j] for j in range(i, min(i + config.conf_batch_size, len(val_ds)))]
            texts = [it["text"] for it in items]
            ys = [it["y"] for it in items]
            logits = conf.forward(texts, device).cpu().numpy().tolist()
            val_logits.extend(logits)
            val_labels.extend(ys)
    if len(val_logits) >= 8 and len(set(val_labels)) > 1:
        raw_probs = 1.0 / (1.0 + np.exp(-np.array(val_logits)))
        raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)
        raw_logits_arr = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
        y_arr = np.array(val_labels).astype(int)
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(raw_logits_arr, y_arr)
        w = float(lr_model.coef_[0][0])
        b = float(lr_model.intercept_[0])
        conf.platt_b = 1.0 / max(abs(w), 1e-6)
        conf.platt_a = -b * conf.platt_b
    else:
        conf.platt_a = 0.0
        conf.platt_b = 1.0
    ece, brier = expected_calibration_error_and_brier(val_logits, val_labels, conf.platt_a, conf.platt_b)
    logger.info(f"[CONF] ECE_after_platt={ece:.4f} Brier={brier:.4f} a={conf.platt_a:.4f} b={conf.platt_b:.4f}")
    return conf


@dataclass
class PrecomputedTransition:
    state_text: str
    state_action_text: str
    next_state_text: str
    r_hat: float
    terminal: float


def precompute_iql_transitions(transitions: List[StepTransition],
                               trajectories_by_id: Dict[str, Trajectory],
                               prm: ProcessRewardModel, config: PRISMConfig,
                               device: torch.device) -> List[PrecomputedTransition]:
    prm.eval()
    state_texts: List[str] = []
    sa_texts: List[str] = []
    next_texts: List[str] = []
    outcomes_at_terminal: List[float] = []
    terminals: List[float] = []
    for tr in transitions:
        trajectory = trajectories_by_id[tr.trajectory_id]
        prefix = trajectory.prefix_text(tr.step_index)
        sa = trajectory.prefix_action_text(tr.step_index)
        nxt = trajectory.prefix_text(tr.step_index + 1) if tr.step_index + 1 < trajectory.length else prefix
        state_texts.append(prefix)
        sa_texts.append(sa)
        next_texts.append(nxt)
        outcomes_at_terminal.append(1.0 if trajectory.success else 0.0)
        terminals.append(1.0 if tr.terminal else 0.0)
    phi_values: List[float] = []
    batch_size = config.prm_batch_size
    for i in range(0, len(sa_texts), batch_size):
        chunk = sa_texts[i:i + batch_size]
        phi = prm.score(chunk, device).cpu().numpy().tolist()
        phi_values.extend(phi)
    precomputed: List[PrecomputedTransition] = []
    for i in range(len(transitions)):
        phi_bar = 2.0 * phi_values[i] - 1.0
        outcome_term = outcomes_at_terminal[i] * terminals[i]
        r_hat = config.alpha_reward_mix * phi_bar + (1.0 - config.alpha_reward_mix) * outcome_term
        precomputed.append(PrecomputedTransition(
            state_text=state_texts[i],
            state_action_text=sa_texts[i],
            next_state_text=next_texts[i],
            r_hat=float(r_hat),
            terminal=float(terminals[i]),
        ))
    return precomputed


class IQLDataset(Dataset):
    def __init__(self, precomputed: List[PrecomputedTransition]):
        self.items = precomputed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "state_text": it.state_text,
            "state_action_text": it.state_action_text,
            "next_state_text": it.next_state_text,
            "r_hat": it.r_hat,
            "terminal": it.terminal,
        }


def collate_iql(batch):
    return {
        "state_texts": [b["state_text"] for b in batch],
        "state_action_texts": [b["state_action_text"] for b in batch],
        "next_state_texts": [b["next_state_text"] for b in batch],
        "r_hat": torch.tensor([b["r_hat"] for b in batch], dtype=torch.float32),
        "terminal": torch.tensor([b["terminal"] for b in batch], dtype=torch.float32),
    }


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff < 0,
                         torch.full_like(diff, tau - 1.0),
                         torch.full_like(diff, tau)).abs()
    return (weight * diff.pow(2)).mean()


def train_iql(value_head: ValueHead, q_head: QValueHead, q_target: QValueHead,
              backbone: EncoderBackbone, precomputed: List[PrecomputedTransition],
              config: PRISMConfig, device: torch.device):
    ds = IQLDataset(precomputed)
    loader = DataLoader(ds, batch_size=config.iql_batch_size, shuffle=True, collate_fn=collate_iql)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    params = list(value_head.parameters()) + list(q_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.iql_lr, weight_decay=0.01)
    total_steps = config.iql_steps
    warmup_steps = int(config.iql_warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    backbone.to(device)
    value_head.to(device)
    q_head.to(device)
    q_target.to(device)
    q_target.load_state_dict(q_head.state_dict())
    for p in q_target.parameters():
        p.requires_grad = False
    step = 0
    loader_iter = iter(loader)
    while step < total_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        with torch.no_grad():
            state_emb = backbone.encode(batch["state_texts"], device)
            sa_emb = backbone.encode(batch["state_action_texts"], device)
            next_emb = backbone.encode(batch["next_state_texts"], device)
        r_hat = batch["r_hat"].to(device)
        terminal = batch["terminal"].to(device)
        v_s = value_head(state_emb)
        with torch.no_grad():
            q_sa_target = q_target(sa_emb)
        loss_v = expectile_loss(q_sa_target - v_s, config.iql_expectile_tau)
        q_sa = q_head(sa_emb)
        with torch.no_grad():
            v_next = value_head(next_emb)
            target = r_hat + config.iql_gamma * (1.0 - terminal) * v_next
        loss_q = F.mse_loss(q_sa, target)
        loss = loss_v + loss_q
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for p_tgt, p_src in zip(q_target.parameters(), q_head.parameters()):
                p_tgt.data.mul_(1.0 - config.iql_polyak).add_(config.iql_polyak * p_src.data)
        step += 1
        if step % config.log_every == 0:
            logger.info(f"[IQL] step={step} loss_v={loss_v.item():.4f} loss_q={loss_q.item():.4f}")
    return value_head, q_head


@dataclass
class AWRSample:
    prompt: str
    action_text: str
    advantage: float


def build_awr_samples(transitions: List[StepTransition], trajectories_by_id: Dict[str, Trajectory],
                      backbone: EncoderBackbone, value_head: ValueHead, q_head: QValueHead,
                      policy: BasePolicy, config: PRISMConfig, device: torch.device) -> List[AWRSample]:
    samples: List[AWRSample] = []
    backbone.eval()
    value_head.eval()
    q_head.eval()
    batch_size = config.iql_batch_size
    with torch.no_grad():
        i = 0
        while i < len(transitions):
            batch = transitions[i:i + batch_size]
            state_texts = []
            sa_texts = []
            prompts = []
            actions = []
            for tr in batch:
                trajectory = trajectories_by_id[tr.trajectory_id]
                prefix = trajectory.prefix_text(tr.step_index)
                state_texts.append(prefix)
                sa_texts.append(trajectory.prefix_action_text(tr.step_index))
                prompts.append(policy.build_prompt(tr.benchmark, prefix))
                actions.append(trajectory.actions[tr.step_index])
            s_emb = backbone.encode(state_texts, device)
            sa_emb = backbone.encode(sa_texts, device)
            v_s = value_head(s_emb).cpu().numpy().tolist()
            q_sa = q_head(sa_emb).cpu().numpy().tolist()
            for p, a, v_val, q_val in zip(prompts, actions, v_s, q_sa):
                samples.append(AWRSample(p, a, float(q_val - v_val)))
            i += batch_size
    return samples


def tokenize_awr_item(prompt: str, action: str, tokenizer, config: PRISMConfig
                      ) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                           max_length=config.policy_max_length)["input_ids"]
    action_ids = tokenizer(action, add_special_tokens=False, truncation=True,
                           max_length=config.action_max_tokens)["input_ids"]
    if tokenizer.eos_token_id is not None:
        action_ids = action_ids + [tokenizer.eos_token_id]
    full_ids = prompt_ids + action_ids
    labels = [-100] * len(prompt_ids) + list(action_ids)
    return full_ids, labels


def collate_awr(batch: List[Dict[str, Any]], tokenizer, config: PRISMConfig):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    weights = []
    pad_id = tokenizer.pad_token_id
    for item in batch:
        full_ids, labels = tokenize_awr_item(item["prompt"], item["action_text"], tokenizer, config)
        attn = [1] * len(full_ids)
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attn, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        weights.append(item["weight"])
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "weights": weights_tensor,
    }


def compute_per_sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    flat_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    flat_loss = flat_loss.view(shift_labels.size())
    valid = (shift_labels != -100).float()
    seq_loss = (flat_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    return seq_loss


def train_awr_policy(policy: BasePolicy, samples: List[AWRSample], config: PRISMConfig, device: torch.device):
    policy.attach_lora()
    model = policy.model
    tokenizer = policy.tokenizer
    model.train()
    beta = config.awr_beta
    delta_max = config.awr_delta_max
    items: List[Dict[str, Any]] = []
    for s in samples:
        w_raw = math.exp(min(beta * s.advantage, beta * delta_max))
        w = w_raw if s.advantage >= 0 else w_raw * config.awr_negative_advantage_weight
        items.append({"prompt": s.prompt, "action_text": s.action_text, "weight": float(w)})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.awr_lr, weight_decay=0.0)
    total_steps = max(1, config.awr_epochs * (len(items) // max(config.awr_batch_size, 1)))
    warmup_steps = int(config.awr_warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    step = 0
    for epoch in range(config.awr_epochs):
        random.shuffle(items)
        for i in range(0, len(items), config.awr_batch_size):
            batch_items = items[i:i + config.awr_batch_size]
            if not batch_items:
                continue
            batch = collate_awr(batch_items, tokenizer, config)
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            weights = batch["weights"].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            seq_losses = compute_per_sequence_loss(outputs.logits, labels)
            loss = (weights * seq_losses).sum() / weights.sum().clamp(min=1e-8)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                logger.info(f"[AWR] epoch={epoch} step={step} loss={loss.item():.4f}")
    model.eval()
    return policy


def self_correcting_inference(policy: BasePolicy, prm: ProcessRewardModel, conf: ConfidenceHead,
                              env: BaseEnv, task_id: str, config: PRISMConfig,
                              device: torch.device) -> Trajectory:
    obs, goal = env.reset(task_id)
    traj = Trajectory(
        trajectory_id=f"infer_{task_id}_{int(time.time()*1000)}",
        benchmark=env.benchmark,
        task_id=task_id,
        goal_text=goal,
    )
    traj.states.append(obs)
    horizon = config.horizon(env.benchmark)
    aborted = False
    for t in range(horizon):
        chosen_action: Optional[str] = None
        reflection: Optional[str] = None
        for retry in range(config.r_max + 1):
            prefix_text = traj.prefix_text(t)
            prompt = policy.build_prompt(env.benchmark, prefix_text, reflection=reflection)
            candidate_generations = policy.sample(prompt, k=config.k_candidates,
                                                  temperature=config.decode_temperature)
            candidate_actions = [policy.parse_action(c) for c in candidate_generations]
            texts_to_score = [prefix_text + "\n[ACT] " + a for a in candidate_actions]
            scores = prm.score(texts_to_score, device).cpu().numpy().tolist()
            best_idx = int(np.argmax(scores))
            best_action = candidate_actions[best_idx]
            conf_text = prefix_text + "\n[ACT] " + best_action
            c_t = float(conf.confidence_calibrated([conf_text], device).cpu().numpy().tolist()[0])
            if c_t >= config.tau_low:
                chosen_action = best_action
                break
            if c_t < config.tau_abort and retry == config.r_max:
                aborted = True
                break
            reflection_prompt = (prefix_text +
                                 "\n\nBriefly identify what is most likely going wrong in this attempt, in two sentences.")
            refl_out = policy.sample(reflection_prompt, k=1, temperature=config.decode_temperature, max_new_tokens=128)
            reflection = refl_out[0].strip().split("\n")[0]
            chosen_action = best_action
        if aborted or chosen_action is None:
            break
        traj.actions.append(chosen_action)
        next_obs, reward, terminal, info = env.step(chosen_action)
        traj.rewards.append(reward)
        traj.states.append(next_obs)
        traj.length += 1
        if terminal:
            traj.success = bool(info.get("success", reward >= env.config.success_threshold(env.benchmark)))
            traj.final_reward = reward
            break
    return traj


def evaluate(policy: BasePolicy, prm: ProcessRewardModel, conf: ConfidenceHead,
             envs: Dict[str, BaseEnv], config: PRISMConfig, device: torch.device,
             max_tasks: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for benchmark, env in envs.items():
        eval_tasks = env.list_eval_tasks()
        if max_tasks is not None:
            eval_tasks = eval_tasks[:max_tasks]
        per_seed_sr: List[float] = []
        per_seed_scores: List[float] = []
        per_seed_steps: List[float] = []
        for seed in config.seeds_eval:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            successes = 0
            scores: List[float] = []
            steps_used: List[int] = []
            for task_id in eval_tasks:
                traj = self_correcting_inference(policy, prm, conf, env, task_id, config, device)
                if benchmark == "alfworld":
                    if traj.success:
                        successes += 1
                        steps_used.append(traj.length)
                    scores.append(1.0 if traj.success else 0.0)
                elif benchmark == "webshop":
                    if traj.final_reward >= config.webshop_success_threshold:
                        successes += 1
                        steps_used.append(traj.length)
                    scores.append(traj.final_reward)
                else:
                    if traj.final_reward >= config.scienceworld_success_threshold:
                        successes += 1
                        steps_used.append(traj.length)
                    scores.append(traj.final_reward)
            per_seed_sr.append(successes / max(len(eval_tasks), 1))
            per_seed_scores.append(float(np.mean(scores)) if scores else 0.0)
            per_seed_steps.append(float(np.mean(steps_used)) if steps_used else 0.0)
        results[benchmark] = {
            "success_rate_mean": float(np.mean(per_seed_sr)),
            "success_rate_std": float(np.std(per_seed_sr)),
            "score_mean": float(np.mean(per_seed_scores)),
            "steps_mean": float(np.mean(per_seed_steps)),
        }
    return results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(config: PRISMConfig, args: argparse.Namespace):
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = {b: build_env(b, config) for b in config.benchmarks}

    logger.info("Stage A: trajectory collection with base policy pi_0")
    base_policy = BasePolicy(config)
    buffer = collect_trajectory_buffer([(base_policy, config.k_collect)], envs, config,
                                       max_tasks_per_env=args.max_tasks_per_env)
    logger.info(f"Collected {len(buffer)} base-policy trajectories")

    if args.run_sft:
        logger.info("Stage A: one-epoch SFT on successful subset to form pi_1")
        successful = [t for t in buffer if t.success]
        if successful:
            sft_samples = []
            for tr in successful:
                for t in range(tr.length):
                    prompt = base_policy.build_prompt(tr.benchmark, tr.prefix_text(t))
                    sft_samples.append(AWRSample(prompt, tr.actions[t], advantage=1.0))
            train_awr_policy(base_policy, sft_samples, config, device)
    finetuned_policy = base_policy

    logger.info("Stage A: pi_1 rollouts to enrich buffer")
    buffer_extra = collect_trajectory_buffer([(finetuned_policy, config.k_collect_finetuned)], envs, config,
                                             max_tasks_per_env=args.max_tasks_per_env)
    buffer.extend(buffer_extra)
    logger.info(f"Total buffer after pi_1 rollouts: {len(buffer)} trajectories")

    logger.info("Stage B1: Monte Carlo step labels")
    triples = monte_carlo_step_labels(buffer, finetuned_policy, envs, config)
    with open(os.path.join(config.output_dir, "step_labels.json"), "w") as f:
        json.dump(triples, f)
    logger.info(f"Generated {len(triples)} step-label triples")

    logger.info("Stage B2: train PRM and confidence head")
    backbone = EncoderBackbone(config.encoder_model_name, max_length=config.encoder_max_length)
    prm = ProcessRewardModel(backbone)
    conf = ConfidenceHead(backbone)
    train_prm(prm, triples, config, device)
    train_confidence_head(conf, buffer, config, device)

    logger.info("Stage B3: train IQL critics with precomputed r_hat")
    trajectories_by_id = {tr.trajectory_id: tr for tr in buffer}
    transitions: List[StepTransition] = []
    for tr in buffer:
        transitions.extend(tr.transitions())
    precomputed = precompute_iql_transitions(transitions, trajectories_by_id, prm, config, device)
    value_head = ValueHead(backbone.hidden_size)
    q_head = QValueHead(backbone.hidden_size)
    q_target = QValueHead(backbone.hidden_size)
    train_iql(value_head, q_head, q_target, backbone, precomputed, config, device)

    logger.info("Stage B4: AWR policy extraction over LoRA")
    awr_samples = build_awr_samples(transitions, trajectories_by_id, backbone, value_head, q_head,
                                    base_policy, config, device)
    train_awr_policy(base_policy, awr_samples, config, device)

    logger.info("Stage C: evaluation with self-correcting inference")
    results = evaluate(base_policy, prm, conf, envs, config, device, max_tasks=args.max_eval_tasks)
    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for benchmark, metrics in results.items():
        logger.info(f"[RESULT] {benchmark}: {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./prism_outputs")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--encoder_model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["alfworld", "webshop", "scienceworld"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tasks_per_env", type=int, default=None)
    parser.add_argument("--max_eval_tasks", type=int, default=None)
    parser.add_argument("--run_sft", action="store_true")
    parser.add_argument("--n_step_labels", type=int, default=60000)
    parser.add_argument("--iql_steps", type=int, default=50000)
    parser.add_argument("--prm_epochs", type=int, default=20)
    parser.add_argument("--awr_epochs", type=int, default=1)
    args = parser.parse_args()
    config = PRISMConfig(
        base_model_name=args.base_model_name,
        encoder_model_name=args.encoder_model_name,
        benchmarks=tuple(args.benchmarks),
        seed=args.seed,
        output_dir=args.output_dir,
        n_step_labels=args.n_step_labels,
        iql_steps=args.iql_steps,
        prm_epochs=args.prm_epochs,
        awr_epochs=args.awr_epochs,
    )
    run_pipeline(config, args)


if __name__ == "__main__":
    main()
