from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from peft import LoraConfig, get_peft_model, TaskType

from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prism")

BENCHMARKS = ("alfworld", "webshop", "scienceworld")


@dataclass
class PRISMConfig:
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    encoder_model_name: str = "microsoft/deberta-v3-base"
    benchmarks: Tuple[str, ...] = BENCHMARKS

    k_collect: int = 4
    k_collect_finetuned: int = 2
    decode_temperature: float = 0.7
    decode_top_p: float = 0.95
    max_new_tokens: int = 256
    reflection_max_new_tokens: int = 128

    n_train_tasks_alfworld: int = 3553
    n_train_tasks_webshop: int = 1000
    n_train_tasks_scienceworld: int = 1371
    n_eval_tasks_alfworld: int = 134
    n_eval_tasks_webshop: int = 500
    n_eval_tasks_scienceworld: int = 270

    n_calib_tasks_alfworld: int = 335
    n_calib_tasks_webshop: int = 250
    n_calib_tasks_scienceworld: int = 250
    calib_rollouts_per_task: int = 2

    webshop_eval_session_offset: int = 0
    webshop_train_session_offset: int = 500

    m_continuations: int = 8
    n_step_labels: int = 60000
    n_step_labels_val: int = 6000
    label_noise_eps: float = 0.0
    min_trajectory_length: int = 3

    lambda_bce: float = 1.0
    lambda_mse: float = 0.5
    prm_epochs: int = 20
    prm_lr: float = 1.5e-5
    prm_batch_size: int = 32
    prm_weight_decay: float = 0.01
    prm_warmup_ratio: float = 0.10
    early_stopping_patience: int = 3

    conf_epochs: int = 20
    conf_lr: float = 1.0e-5
    conf_batch_size: int = 32
    conf_weight_decay: float = 0.01
    conf_warmup_ratio: float = 0.10
    conf_samples_per_traj: int = 3

    iql_expectile_tau: float = 0.7
    iql_gamma: float = 0.99
    iql_polyak: float = 0.005
    iql_lr: float = 5.0e-5
    iql_batch_size: int = 64
    iql_steps: int = 50000
    iql_warmup_ratio: float = 0.10
    iql_weight_decay: float = 0.01
    alpha_reward_mix: float = 0.7

    awr_beta: float = 3.0
    awr_delta_max: float = 4.0
    awr_negative_advantage_weight: float = 0.1
    awr_steps: int = 50000
    awr_lr: float = 1.0e-4
    awr_batch_size: int = 16
    awr_warmup_ratio: float = 0.05
    awr_weight_decay: float = 0.0
    sft_epochs: int = 1

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0

    k_candidates: int = 3
    tau_low: float = 0.45
    tau_abort: float = 0.20
    r_max: int = 2
    tau_low_grid: Tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    tau_abort_grid: Tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
    threshold_search_tasks: int = 64

    horizon_alfworld: int = 30
    horizon_webshop: int = 25
    horizon_scienceworld: int = 30

    webshop_success_threshold: float = 0.9
    scienceworld_success_threshold: float = 90.0

    encoder_max_length: int = 512
    encoder_goal_max_tokens: int = 128
    encoder_action_max_tokens: int = 64
    policy_max_length: int = 4096
    action_max_tokens: int = 256

    seed: int = 42
    seeds_eval: Tuple[int, ...] = (1, 7, 13, 23, 42)
    output_dir: str = "./prism_outputs"
    log_every: int = 100
    save_every: int = 5000
    bf16: bool = True
    gradient_checkpointing: bool = True
    ece_bins: int = 10

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

    def n_train_tasks(self, benchmark: str) -> int:
        return {
            "alfworld": self.n_train_tasks_alfworld,
            "webshop": self.n_train_tasks_webshop,
            "scienceworld": self.n_train_tasks_scienceworld,
        }[benchmark]

    def n_eval_tasks(self, benchmark: str) -> int:
        return {
            "alfworld": self.n_eval_tasks_alfworld,
            "webshop": self.n_eval_tasks_webshop,
            "scienceworld": self.n_eval_tasks_scienceworld,
        }[benchmark]

    def n_calib_tasks(self, benchmark: str) -> int:
        return {
            "alfworld": self.n_calib_tasks_alfworld,
            "webshop": self.n_calib_tasks_webshop,
            "scienceworld": self.n_calib_tasks_scienceworld,
        }[benchmark]

    def primary_metric(self, benchmark: str) -> str:
        return "mean_reward" if benchmark == "scienceworld" else "success_rate"


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
    aborted: bool = False
    decoded_tokens: int = 0
    reflection_steps: int = 0

    def transitions(self) -> List[StepTransition]:
        return [
            StepTransition(
                benchmark=self.benchmark,
                step_index=t,
                trajectory_id=self.trajectory_id,
                env_reward=self.rewards[t],
                terminal=(t == self.length - 1),
            )
            for t in range(self.length)
        ]

    def history_text(self, t: int) -> str:
        parts: List[str] = []
        for i in range(min(t, self.length)):
            parts.append(f"Obs {i}: {self.states[i]}")
            parts.append(f"Act {i}: {self.actions[i]}")
        if t < len(self.states):
            parts.append(f"Obs {t}: {self.states[t]}")
        return "\n".join(parts)

    def prefix_text(self, t: int) -> str:
        return f"Goal: {self.goal_text}\n" + self.history_text(t)

    def action_at(self, t: int) -> str:
        return self.actions[t] if t < len(self.actions) else ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "Trajectory":
        return Trajectory(**payload)


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

REFLECTION_INSTRUCTION = (
    "Briefly identify what is most likely going wrong in this attempt, in two sentences."
)


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

    def close(self) -> None:
        return None


def extract_alfworld_goal(observation: str, fallback: str) -> str:
    marker = "Your task is to:"
    if marker in observation:
        return observation.split(marker, 1)[1].strip().split("\n")[0].strip()
    return fallback


class AlfWorldEnv(BaseEnv):
    def __init__(self, benchmark: str, config: PRISMConfig):
        super().__init__(benchmark, config)
        import alfworld.agents.environment as alf_env
        import alfworld.agents.modules.generic as generic

        self._cfg = generic.load_config()
        self._train_wrapper = alf_env.AlfredTWEnv(self._cfg, train_eval="train")
        self._eval_wrapper = alf_env.AlfredTWEnv(self._cfg, train_eval="eval_out_of_distribution")
        train_files = self._gamefiles(self._train_wrapper)
        eval_files = self._gamefiles(self._eval_wrapper)
        if not train_files or not eval_files:
            raise RuntimeError(
                "ALFWorld game files were not found. Run `alfworld-download` and make sure "
                "ALFWORLD_DATA points at the downloaded task suite."
            )
        train_files = train_files[: config.n_train_tasks_alfworld]
        eval_files = eval_files[: config.n_eval_tasks_alfworld]
        self._train_index = {f"alfworld_train_{i}": gf for i, gf in enumerate(train_files)}
        self._eval_index = {f"alfworld_eval_{i}": gf for i, gf in enumerate(eval_files)}
        self._active = None

    @staticmethod
    def _gamefiles(wrapper) -> List[str]:
        files = getattr(wrapper, "game_files", None)
        if files is None:
            files = getattr(wrapper, "gamefiles", None)
        return sorted(str(gf) for gf in files) if files else []

    def list_train_tasks(self) -> List[str]:
        return list(self._train_index.keys())

    def list_eval_tasks(self) -> List[str]:
        return list(self._eval_index.keys())

    def _wrapper_for(self, task_id: str):
        if task_id in self._eval_index:
            return self._eval_wrapper, self._eval_index[task_id]
        if task_id in self._train_index:
            return self._train_wrapper, self._train_index[task_id]
        raise KeyError(f"Unknown ALFWorld task id {task_id}")

    def reset(self, task_id: str) -> Tuple[str, str]:
        wrapper, gamefile = self._wrapper_for(task_id)
        wrapper.game_files = [gamefile]
        wrapper.num_games = 1
        self._active = wrapper.init_env(batch_size=1)
        obs, _info = self._active.reset()
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
        return obs_text, extract_alfworld_goal(obs_text, task_id)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self._active.step([action])
        next_obs = obs[0] if isinstance(obs, (list, tuple)) else str(obs)
        reward = float(scores[0]) if hasattr(scores, "__len__") else float(scores)
        done = bool(dones[0]) if hasattr(dones, "__len__") else bool(dones)
        if isinstance(infos, (list, tuple)) and infos and isinstance(infos[0], dict):
            info = dict(infos[0])
        elif isinstance(infos, dict):
            info = {k: (v[0] if isinstance(v, (list, tuple)) and v else v) for k, v in infos.items()}
        else:
            info = {}
        won = info.get("won", None)
        info["success"] = bool(won) if won is not None else bool(reward >= 1.0)
        info["score"] = 1.0 if info["success"] else 0.0
        return next_obs, reward, done, info


class ScienceWorldEnv(BaseEnv):
    def __init__(self, benchmark: str, config: PRISMConfig):
        super().__init__(benchmark, config)
        from scienceworld import ScienceWorldEnv as SWEnv

        self._env = SWEnv("", "", envStepLimit=config.horizon_scienceworld)
        self._task_names = list(self._env.getTaskNames())
        self._train_tasks = self._build_split(True, config.n_train_tasks_scienceworld)
        self._eval_tasks = self._build_split(False, config.n_eval_tasks_scienceworld)
        self._score = 0.0

    def _build_split(self, train: bool, limit: int) -> List[str]:
        pools: List[List[str]] = []
        for name in self._task_names:
            self._env.load(name, 0, "easy", generateGoldPath=False)
            variations = self._env.getVariationsTrain() if train else self._env.getVariationsTest()
            pools.append([f"{name}::{int(v)}" for v in variations])
        tasks: List[str] = []
        cursor = 0
        while len(tasks) < limit and any(cursor < len(pool) for pool in pools):
            for pool in pools:
                if cursor < len(pool) and len(tasks) < limit:
                    tasks.append(pool[cursor])
            cursor += 1
        return tasks

    def list_train_tasks(self) -> List[str]:
        return list(self._train_tasks)

    def list_eval_tasks(self) -> List[str]:
        return list(self._eval_tasks)

    def reset(self, task_id: str) -> Tuple[str, str]:
        name, variation = task_id.rsplit("::", 1)
        self._env.load(name, int(variation), "easy", generateGoldPath=False)
        obs, _info = self._env.reset()
        self._score = 0.0
        return str(obs), str(self._env.getTaskDescription())

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, is_completed, _info = self._env.step(action)
        self._score = max(self._score, float(reward))
        success = bool(is_completed) and self._score >= self.config.scienceworld_success_threshold
        return str(obs), float(reward), bool(is_completed), {
            "success": success,
            "score": self._score,
            "raw_reward": float(reward),
        }


class WebShopEnv(BaseEnv):
    def __init__(self, benchmark: str, config: PRISMConfig):
        super().__init__(benchmark, config)
        try:
            from web_agent_site.envs import WebAgentTextEnv
        except ImportError as exc:
            raise ImportError(
                "WebShop is not installed. Clone https://github.com/princeton-nlp/WebShop, run "
                "setup.sh, then add the WebShop repository root to PYTHONPATH."
            ) from exc
        self._env = WebAgentTextEnv(observation_mode="text", num_products=None)
        self._score = 0.0

    def list_train_tasks(self) -> List[str]:
        start = self.config.webshop_train_session_offset
        return [f"webshop_train_{start + i}" for i in range(self.config.n_train_tasks_webshop)]

    def list_eval_tasks(self) -> List[str]:
        start = self.config.webshop_eval_session_offset
        return [f"webshop_eval_{start + i}" for i in range(self.config.n_eval_tasks_webshop)]

    def reset(self, task_id: str) -> Tuple[str, str]:
        session = int(task_id.rsplit("_", 1)[-1])
        obs, info = self._env.reset(session=session)
        self._score = 0.0
        goal = info.get("instruction_text", str(task_id)) if isinstance(info, dict) else str(task_id)
        return str(obs), str(goal)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, done, _info = self._env.step(action)
        self._score = max(self._score, float(reward))
        success = bool(done) and self._score >= self.config.webshop_success_threshold
        return str(obs), float(reward), bool(done), {
            "success": success,
            "score": self._score,
            "raw_reward": float(reward),
        }


def build_env(benchmark: str, config: PRISMConfig) -> BaseEnv:
    if benchmark == "alfworld":
        return AlfWorldEnv(benchmark, config)
    if benchmark == "scienceworld":
        return ScienceWorldEnv(benchmark, config)
    if benchmark == "webshop":
        return WebShopEnv(benchmark, config)
    raise ValueError(f"Unknown benchmark {benchmark}")


class BasePolicy:
    def __init__(self, config: PRISMConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        dtype = torch.bfloat16 if (config.bf16 and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.config.use_cache = True
        self.adapters: List[str] = []
        self.decoded_tokens = 0

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=list(self.config.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
        )

    def add_adapter(self, name: str) -> None:
        if name in self.adapters:
            self.model.set_adapter(name)
            return
        if not self.adapters:
            self.model = get_peft_model(self.model, self._lora_config(), adapter_name=name)
        else:
            self.model.add_adapter(name, self._lora_config())
        self.adapters.append(name)
        self.model.set_adapter(name)

    def set_adapter(self, name: str) -> None:
        if name not in self.adapters:
            raise KeyError(f"Adapter {name} has not been created")
        self.model.set_adapter(name)

    @contextlib.contextmanager
    def using(self, adapter: Optional[str]):
        if not self.adapters:
            yield
            return
        previous = getattr(self.model, "active_adapter", None)
        if adapter is None:
            with self.model.disable_adapter():
                yield
            return
        self.model.set_adapter(adapter)
        try:
            yield
        finally:
            if isinstance(previous, str) and previous in self.adapters:
                self.model.set_adapter(previous)

    def enable_training_mode(self) -> None:
        self.model.config.use_cache = False
        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        self.model.train()

    def enable_inference_mode(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        self.model.eval()

    def build_prompt(self, benchmark: str, prefix_text: str,
                     reflection: Optional[str] = None) -> str:
        parts = [REACT_SYSTEM_PROMPT, "", EXEMPLARS[benchmark], "", prefix_text]
        if reflection:
            parts.extend(["", f"Reflection: {reflection}"])
        parts.append("Thought:")
        return "\n".join(parts)

    def build_reflection_prompt(self, benchmark: str, prefix_text: str,
                                candidates: List[str], scores: List[float]) -> str:
        table = ["Candidate actions and process-reward scores:"]
        for rank, (action, score) in enumerate(zip(candidates, scores), start=1):
            table.append(f"{rank}. {action}\tPRM={score:.3f}")
        return "\n".join([REACT_SYSTEM_PROMPT, "", EXEMPLARS[benchmark], "", prefix_text, "",
                          "\n".join(table), "", REFLECTION_INSTRUCTION])

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
        generated = out[:, inputs["input_ids"].shape[1]:]
        self.decoded_tokens += int((generated != self.tokenizer.pad_token_id).sum().item())
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]

    @staticmethod
    def parse_action(generated_text: str) -> str:
        if "Action:" in generated_text:
            return generated_text.split("Action:", 1)[1].split("\n")[0].strip()
        return generated_text.strip().split("\n")[0].strip()


def rollout_one_trajectory(policy: BasePolicy, env: BaseEnv, task_id: str,
                           config: PRISMConfig, adapter: Optional[str] = None,
                           tag: str = "roll") -> Optional[Trajectory]:
    horizon = config.horizon(env.benchmark)
    obs, goal = env.reset(task_id)
    traj = Trajectory(
        trajectory_id=f"{tag}_{env.benchmark}_{task_id}_{int(time.time() * 1e6)}_{random.randint(0, 999999)}",
        benchmark=env.benchmark,
        task_id=task_id,
        goal_text=goal,
    )
    traj.states.append(obs)
    tokens_before = policy.decoded_tokens
    last_score = 0.0
    with policy.using(adapter):
        for t in range(horizon):
            prompt = policy.build_prompt(env.benchmark, traj.prefix_text(t))
            samples = policy.sample(prompt, k=1, temperature=config.decode_temperature)
            action = policy.parse_action(samples[0])
            traj.actions.append(action)
            next_obs, reward, terminal, info = env.step(action)
            traj.rewards.append(reward)
            traj.states.append(next_obs)
            traj.length += 1
            last_score = float(info.get("score", reward))
            if terminal:
                traj.success = bool(info.get(
                    "success", reward >= config.success_threshold(env.benchmark)))
                break
    traj.final_reward = last_score
    traj.decoded_tokens = policy.decoded_tokens - tokens_before
    return traj if traj.length > 0 else None


def collect_trajectories(policy: BasePolicy, envs: Dict[str, BaseEnv],
                         tasks_by_benchmark: Dict[str, List[str]], n_rollouts: int,
                         config: PRISMConfig, adapter: Optional[str] = None,
                         tag: str = "roll") -> List[Trajectory]:
    buffer: List[Trajectory] = []
    for benchmark, env in envs.items():
        tasks = tasks_by_benchmark.get(benchmark, [])
        for index, task_id in enumerate(tasks):
            for _ in range(n_rollouts):
                traj = rollout_one_trajectory(policy, env, task_id, config,
                                              adapter=adapter, tag=tag)
                if traj is not None:
                    buffer.append(traj)
            if (index + 1) % config.log_every == 0:
                logger.info(f"[{tag}] {benchmark}: {index + 1}/{len(tasks)} tasks rolled out")
    return buffer


def filter_and_deduplicate(buffer: List[Trajectory], config: PRISMConfig) -> List[Trajectory]:
    seen = set()
    kept: List[Trajectory] = []
    for traj in buffer:
        if traj.length < config.min_trajectory_length:
            continue
        key = (traj.benchmark, traj.task_id, tuple(traj.actions))
        if key in seen:
            continue
        seen.add(key)
        kept.append(traj)
    return kept


def summarize_buffer(buffer: List[Trajectory]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for benchmark in sorted({t.benchmark for t in buffer}):
        subset = [t for t in buffer if t.benchmark == benchmark]
        summary[benchmark] = {
            "tasks": float(len({t.task_id for t in subset})),
            "trajectories": float(len(subset)),
            "success_rate": float(np.mean([t.success for t in subset])),
            "avg_length": float(np.mean([t.length for t in subset])),
        }
    return summary


def continue_rollout_from_prefix(parent: Trajectory, t: int, policy: BasePolicy,
                                 env: BaseEnv, config: PRISMConfig,
                                 adapter: Optional[str]) -> bool:
    horizon = config.horizon(parent.benchmark)
    replay_actions = parent.actions[: t + 1]
    if not replay_actions:
        return bool(parent.success)
    env.reset(parent.task_id)
    for action in replay_actions:
        _obs, reward, terminal, info = env.step(action)
        if terminal:
            return bool(info.get(
                "success", reward >= config.success_threshold(parent.benchmark)))
    cur = Trajectory(
        trajectory_id=f"{parent.trajectory_id}_cont_{random.randint(0, 9999999)}",
        benchmark=parent.benchmark,
        task_id=parent.task_id,
        goal_text=parent.goal_text,
        states=list(parent.states[: t + 2]),
        actions=list(replay_actions),
        rewards=list(parent.rewards[: t + 1]),
        length=t + 1,
    )
    with policy.using(adapter):
        for _ in range(max(horizon - cur.length, 0)):
            prompt = policy.build_prompt(cur.benchmark, cur.prefix_text(cur.length))
            samples = policy.sample(prompt, k=1, temperature=config.decode_temperature)
            action = policy.parse_action(samples[0])
            cur.actions.append(action)
            next_obs, reward, terminal, info = env.step(action)
            cur.states.append(next_obs)
            cur.rewards.append(reward)
            cur.length += 1
            if terminal:
                return bool(info.get(
                    "success", reward >= config.success_threshold(cur.benchmark)))
    return False


def sample_prefix_pool(buffer: List[Trajectory], config: PRISMConfig,
                       rng: random.Random) -> List[Tuple[Trajectory, int]]:
    by_benchmark: Dict[str, List[Tuple[Trajectory, int]]] = {}
    for traj in buffer:
        pool = by_benchmark.setdefault(traj.benchmark, [])
        for t in range(traj.length):
            pool.append((traj, t))
    benchmarks = sorted(by_benchmark.keys())
    if not benchmarks:
        return []
    quota = config.n_step_labels // len(benchmarks)
    sampled: List[Tuple[Trajectory, int]] = []
    for benchmark in benchmarks:
        pool = by_benchmark[benchmark]
        succ = [p for p in pool if p[0].success]
        fail = [p for p in pool if not p[0].success]
        take_succ = min(quota // 2, len(succ))
        take_fail = min(quota - take_succ, len(fail))
        take_succ = min(quota - take_fail, len(succ))
        sampled.extend(rng.sample(succ, take_succ) + rng.sample(fail, take_fail))
    rng.shuffle(sampled)
    return sampled


def monte_carlo_step_labels(buffer: List[Trajectory], policy: BasePolicy,
                            envs: Dict[str, BaseEnv], config: PRISMConfig,
                            adapter: Optional[str]) -> List[Dict[str, Any]]:
    rng = random.Random(config.seed)
    sampled = sample_prefix_pool(buffer, config, rng)
    logger.info(f"[B1] labeling {len(sampled)} prefixes with M={config.m_continuations} continuations")
    triples: List[Dict[str, Any]] = []
    for index, (traj, t) in enumerate(sampled):
        env = envs[traj.benchmark]
        successes = sum(
            1 for _ in range(config.m_continuations)
            if continue_rollout_from_prefix(traj, t, policy, env, config, adapter)
        )
        q_hat = successes / float(config.m_continuations)
        triples.append({
            "goal_text": traj.goal_text,
            "history_text": traj.history_text(t),
            "action_text": traj.action_at(t),
            "q_hat": q_hat,
            "y_label": int(q_hat > 0.5),
            "trajectory_id": traj.trajectory_id,
            "step_index": t,
            "benchmark": traj.benchmark,
            "outcome": int(traj.success),
        })
        if (index + 1) % config.log_every == 0:
            logger.info(f"[B1] labeled {index + 1}/{len(sampled)} prefixes")
    return triples


def corrupt_step_labels(triples: List[Dict[str, Any]], eps: float,
                        seed: int) -> List[Dict[str, Any]]:
    if eps <= 0.0:
        return triples
    rng = random.Random(seed)
    corrupted = 0
    for item in triples:
        if rng.random() < eps:
            item["q_hat"] = 1.0 - float(item["q_hat"])
            item["y_label"] = int(item["q_hat"] > 0.5)
            corrupted += 1
    logger.info(f"[B1] corrupted {corrupted}/{len(triples)} step labels at eps={eps}")
    return triples


class EncoderBackbone(nn.Module):
    def __init__(self, config: PRISMConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.max_length = config.encoder_max_length
        self.goal_max_tokens = config.encoder_goal_max_tokens
        self.action_max_tokens = config.encoder_action_max_tokens
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id or 0

    def _ids(self, text: str, limit: Optional[int] = None) -> List[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids[:limit] if limit is not None else ids

    def build_input_ids(self, goal: str, history: str, action: str) -> List[int]:
        goal_ids = self._ids(f"Goal: {goal}", self.goal_max_tokens)
        action_ids = self._ids(f"[ACT] {action}", self.action_max_tokens) if action else []
        n_special = int(self.cls_id is not None) + 2 * int(self.sep_id is not None)
        budget = max(self.max_length - n_special - len(goal_ids) - len(action_ids), 0)
        history_ids = self._ids(history)
        if len(history_ids) > budget:
            head = budget // 2
            tail = budget - head
            history_ids = history_ids[:head] + (history_ids[-tail:] if tail > 0 else [])
        ids: List[int] = []
        if self.cls_id is not None:
            ids.append(self.cls_id)
        ids.extend(goal_ids)
        if self.sep_id is not None:
            ids.append(self.sep_id)
        ids.extend(history_ids)
        ids.extend(action_ids)
        if self.sep_id is not None:
            ids.append(self.sep_id)
        return ids[: self.max_length]

    def encode_triples(self, items: List[Tuple[str, str, str]],
                       device: torch.device) -> torch.Tensor:
        batch_ids = [self.build_input_ids(goal, history, action) for goal, history, action in items]
        width = max(len(ids) for ids in batch_ids)
        input_ids = torch.full((len(batch_ids), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), width), dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class ProcessRewardModel(nn.Module):
    def __init__(self, backbone: EncoderBackbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.hidden_size, 1)

    def forward(self, items: List[Tuple[str, str, str]], device: torch.device) -> torch.Tensor:
        return self.head(self.backbone.encode_triples(items, device)).squeeze(-1)

    @torch.no_grad()
    def score(self, items: List[Tuple[str, str, str]], device: torch.device) -> torch.Tensor:
        return torch.sigmoid(self.forward(items, device))


class ConfidenceHead(nn.Module):
    def __init__(self, backbone: EncoderBackbone, mlp_hidden: int = 256):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(backbone.hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 1)
        self.platt: Dict[str, Tuple[float, float]] = {}

    def forward(self, items: List[Tuple[str, str, str]], device: torch.device) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(self.backbone.encode_triples(items, device)))).squeeze(-1)

    @torch.no_grad()
    def confidence_raw(self, items: List[Tuple[str, str, str]],
                       device: torch.device) -> torch.Tensor:
        return torch.sigmoid(self.forward(items, device))

    def set_platt(self, benchmark: str, a: float, b: float) -> None:
        self.platt[benchmark] = (float(a), float(b))

    def get_platt(self, benchmark: str) -> Optional[Tuple[float, float]]:
        return self.platt.get(benchmark, self.platt.get("global"))

    @torch.no_grad()
    def confidence_calibrated(self, items: List[Tuple[str, str, str]], benchmark: str,
                              device: torch.device) -> torch.Tensor:
        raw = self.confidence_raw(items, device).clamp(1e-6, 1 - 1e-6)
        params = self.get_platt(benchmark)
        if params is None:
            return raw
        a, b = params
        logits = torch.log(raw / (1 - raw))
        return torch.sigmoid((logits - a) / (b if abs(b) > 1e-6 else 1e-6))


class ScalarHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings).squeeze(-1)


def triple_of(item: Dict[str, Any]) -> Tuple[str, str, str]:
    return item["goal_text"], item["history_text"], item["action_text"]


class PRMDataset(Dataset):
    def __init__(self, triples: List[Dict[str, Any]]):
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.triples[idx]
        return {
            "triple": triple_of(item),
            "q_hat": float(item["q_hat"]),
            "y_label": float(item["y_label"]),
        }


def collate_prm(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "items": [b["triple"] for b in batch],
        "q_hat": torch.tensor([b["q_hat"] for b in batch], dtype=torch.float32),
        "y_label": torch.tensor([b["y_label"] for b in batch], dtype=torch.float32),
    }


class ConfidenceDataset(Dataset):
    def __init__(self, buffer: List[Trajectory], samples_per_traj: int, seed: int):
        rng = random.Random(seed)
        self.items: List[Tuple[Tuple[str, str, str], float]] = []
        for traj in buffer:
            if traj.length < 1:
                continue
            for _ in range(samples_per_traj):
                t = rng.randint(1, traj.length) - 1
                self.items.append((
                    (traj.goal_text, traj.history_text(t), traj.action_at(t)),
                    float(traj.success),
                ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        triple, y = self.items[idx]
        return {"triple": triple, "y": y}


def collate_conf(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "items": [b["triple"] for b in batch],
        "y": torch.tensor([b["y"] for b in batch], dtype=torch.float32),
    }


def build_optimizer(params: Iterable[nn.Parameter], lr: float, weight_decay: float,
                    config: PRISMConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                             betas=(config.adam_beta1, config.adam_beta2))


def train_prm(prm: ProcessRewardModel, triples: List[Dict[str, Any]],
              config: PRISMConfig, device: torch.device) -> Dict[str, Any]:
    shuffled = list(triples)
    random.Random(config.seed).shuffle(shuffled)
    n_val = min(config.n_step_labels_val, max(1, len(shuffled) // 10))
    val_triples, train_triples = shuffled[:n_val], shuffled[n_val:]
    logger.info(f"[B2][PRM] split: {len(train_triples)} train / {len(val_triples)} validation")
    train_loader = DataLoader(PRMDataset(train_triples), batch_size=config.prm_batch_size,
                              shuffle=True, collate_fn=collate_prm)
    val_loader = DataLoader(PRMDataset(val_triples), batch_size=config.prm_batch_size,
                            shuffle=False, collate_fn=collate_prm)
    optimizer = build_optimizer(prm.parameters(), config.prm_lr, config.prm_weight_decay, config)
    total_steps = max(1, config.prm_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.prm_warmup_ratio * total_steps), total_steps)
    prm.to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    best_val, best_state, patience = float("inf"), None, 0
    history: List[Dict[str, float]] = []
    step = 0
    for epoch in range(config.prm_epochs):
        prm.train()
        for batch in train_loader:
            logits = prm.forward(batch["items"], device)
            loss = (config.lambda_bce * bce_fn(logits, batch["y_label"].to(device))
                    + config.lambda_mse * F.mse_loss(torch.sigmoid(logits),
                                                     batch["q_hat"].to(device)))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prm.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                logger.info(f"[B2][PRM] epoch={epoch} step={step} loss={loss.item():.4f}")
        prm.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits = prm.forward(batch["items"], device)
                probs = torch.sigmoid(logits)
                y = batch["y_label"].to(device)
                batch_loss = (config.lambda_bce * bce_fn(logits, y)
                              + config.lambda_mse * F.mse_loss(probs, batch["q_hat"].to(device)))
                val_loss += float(batch_loss.item()) * int(y.shape[0])
                correct += int(((probs > 0.5).float() == y).sum().item())
                total += int(y.shape[0])
        val_loss /= max(total, 1)
        accuracy = correct / max(total, 1)
        history.append({"epoch": float(epoch), "val_loss": val_loss, "step_label_acc": accuracy})
        logger.info(f"[B2][PRM] epoch={epoch} val_loss={val_loss:.4f} step_label_acc={accuracy:.4f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in prm.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                logger.info(f"[B2][PRM] early stopping at epoch {epoch}")
                break
    if best_state is not None:
        prm.load_state_dict(best_state)
        prm.to(device)
    return {"history": history, "best_val_loss": best_val}


def train_confidence_head(conf: ConfidenceHead, train_buffer: List[Trajectory],
                          config: PRISMConfig, device: torch.device) -> Dict[str, Any]:
    dataset = ConfidenceDataset(train_buffer, config.conf_samples_per_traj, config.seed)
    indices = list(range(len(dataset)))
    random.Random(config.seed).shuffle(indices)
    n_val = max(1, len(dataset) // 10)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, indices[n_val:]),
                              batch_size=config.conf_batch_size, shuffle=True,
                              collate_fn=collate_conf)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, indices[:n_val]),
                            batch_size=config.conf_batch_size, shuffle=False,
                            collate_fn=collate_conf)
    optimizer = build_optimizer(conf.parameters(), config.conf_lr, config.conf_weight_decay, config)
    total_steps = max(1, config.conf_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.conf_warmup_ratio * total_steps), total_steps)
    conf.to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    best_val, best_state, patience = float("inf"), None, 0
    step = 0
    for epoch in range(config.conf_epochs):
        conf.train()
        for batch in train_loader:
            loss = bce_fn(conf.forward(batch["items"], device), batch["y"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conf.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % config.log_every == 0:
                logger.info(f"[B2][CONF] epoch={epoch} step={step} loss={loss.item():.4f}")
        conf.eval()
        val_loss, total = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(device)
                val_loss += float(bce_fn(conf.forward(batch["items"], device), y).item()) * int(y.shape[0])
                total += int(y.shape[0])
        val_loss /= max(total, 1)
        logger.info(f"[B2][CONF] epoch={epoch} val_loss={val_loss:.4f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in conf.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                logger.info(f"[B2][CONF] early stopping at epoch {epoch}")
                break
    if best_state is not None:
        conf.load_state_dict(best_state)
        conf.to(device)
    return {"best_val_loss": best_val}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, max(len(probs), 1)
    for m in range(n_bins):
        lo, hi = edges[m], edges[m + 1]
        mask = (probs >= lo) & (probs < hi) if m < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels) ** 2).mean()) if len(probs) else 0.0


def fit_platt_scalars(raw_probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(logits, labels.astype(int))
    w = float(model.coef_[0][0])
    c = float(model.intercept_[0])
    if abs(w) < 1e-6:
        return 0.0, 1.0
    b = 1.0 / w
    return -c * b, b


def apply_platt(raw_probs: np.ndarray, a: float, b: float) -> np.ndarray:
    clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-(logits - a) / (b if abs(b) > 1e-6 else 1e-6)))


@torch.no_grad()
def confidence_on_pool(conf: ConfidenceHead, pool: List[Trajectory], config: PRISMConfig,
                       device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(config.seed)
    items: List[Tuple[str, str, str]] = []
    labels: List[float] = []
    for traj in pool:
        if traj.length < 1:
            continue
        t = rng.randint(1, traj.length) - 1
        items.append((traj.goal_text, traj.history_text(t), traj.action_at(t)))
        labels.append(float(traj.success))
    conf.eval()
    probs: List[float] = []
    for i in range(0, len(items), config.conf_batch_size):
        chunk = items[i:i + config.conf_batch_size]
        probs.extend(conf.confidence_raw(chunk, device).cpu().numpy().tolist())
    return np.array(probs), np.array(labels)


def calibrate_confidence(conf: ConfidenceHead, calibration_pool: List[Trajectory],
                         config: PRISMConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    pooled_probs: List[np.ndarray] = []
    pooled_labels: List[np.ndarray] = []
    for benchmark in sorted({t.benchmark for t in calibration_pool}):
        subset = [t for t in calibration_pool if t.benchmark == benchmark]
        probs, labels = confidence_on_pool(conf, subset, config, device)
        if len(probs) < 8 or len(set(labels.tolist())) < 2:
            logger.warning(f"[B2][CALIB] insufficient calibration data for {benchmark}")
            continue
        pooled_probs.append(probs)
        pooled_labels.append(labels)
        ece_before = expected_calibration_error(probs, labels, config.ece_bins)
        brier_before = brier_score(probs, labels)
        a, b = fit_platt_scalars(probs, labels)
        conf.set_platt(benchmark, a, b)
        calibrated = apply_platt(probs, a, b)
        ece_after = expected_calibration_error(calibrated, labels, config.ece_bins)
        brier_after = brier_score(calibrated, labels)
        report[benchmark] = {
            "n_trajectories": float(len(subset)),
            "platt_a": a,
            "platt_b": b,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
        }
        logger.info(f"[B2][CALIB] {benchmark}: n={len(subset)} "
                    f"ECE {ece_before:.4f} -> {ece_after:.4f} "
                    f"Brier {brier_before:.4f} -> {brier_after:.4f} (a={a:.4f}, b={b:.4f})")
    if pooled_probs:
        probs = np.concatenate(pooled_probs)
        labels = np.concatenate(pooled_labels)
        a, b = fit_platt_scalars(probs, labels)
        conf.set_platt("global", a, b)
        report["global"] = {
            "platt_a": a,
            "platt_b": b,
            "ece_before": expected_calibration_error(probs, labels, config.ece_bins),
            "ece_after": expected_calibration_error(apply_platt(probs, a, b), labels, config.ece_bins),
        }
    return report


@dataclass
class PrecomputedTransition:
    benchmark: str
    state_key: int
    state_action_key: int
    next_state_key: int
    r_hat: float
    terminal: float


def precompute_iql_transitions(transitions: List[StepTransition],
                               trajectories_by_id: Dict[str, Trajectory],
                               prm: ProcessRewardModel, backbone: EncoderBackbone,
                               config: PRISMConfig, device: torch.device
                               ) -> Tuple[List[PrecomputedTransition], torch.Tensor]:
    unique: Dict[Tuple[str, str, str], int] = {}
    keys: List[Tuple[int, int, int]] = []
    metadata: List[Tuple[str, float, float]] = []

    def key_for(triple: Tuple[str, str, str]) -> int:
        if triple not in unique:
            unique[triple] = len(unique)
        return unique[triple]

    for tr in transitions:
        traj = trajectories_by_id[tr.trajectory_id]
        t = tr.step_index
        keys.append((
            key_for((traj.goal_text, traj.history_text(t), "")),
            key_for((traj.goal_text, traj.history_text(t), traj.action_at(t))),
            key_for((traj.goal_text, traj.history_text(min(t + 1, traj.length)), "")),
        ))
        metadata.append((tr.benchmark, 1.0 if traj.success else 0.0, 1.0 if tr.terminal else 0.0))

    ordered = [triple for triple, _ in sorted(unique.items(), key=lambda kv: kv[1])]
    logger.info(f"[B3] encoding {len(ordered)} unique states for {len(transitions)} transitions")
    prm.eval()
    backbone.eval()
    embeddings = torch.zeros(len(ordered), backbone.hidden_size, dtype=torch.float32)
    phi = torch.zeros(len(ordered), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(ordered), config.prm_batch_size):
            chunk = ordered[i:i + config.prm_batch_size]
            emb = backbone.encode_triples(chunk, device)
            embeddings[i:i + len(chunk)] = emb.detach().float().cpu()
            phi[i:i + len(chunk)] = torch.sigmoid(prm.head(emb).squeeze(-1)).detach().float().cpu()

    precomputed: List[PrecomputedTransition] = []
    for (s_key, sa_key, ns_key), (benchmark, outcome, terminal) in zip(keys, metadata):
        phi_bar = 2.0 * float(phi[sa_key]) - 1.0
        r_hat = (config.alpha_reward_mix * phi_bar
                 + (1.0 - config.alpha_reward_mix) * outcome * terminal)
        precomputed.append(PrecomputedTransition(
            benchmark=benchmark,
            state_key=s_key,
            state_action_key=sa_key,
            next_state_key=ns_key,
            r_hat=float(r_hat),
            terminal=float(terminal),
        ))
    return precomputed, embeddings


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff < 0, torch.full_like(diff, 1.0 - tau), torch.full_like(diff, tau))
    return (weight * diff.pow(2)).mean()


def train_iql(value_head: ScalarHead, q_head: ScalarHead, q_target: ScalarHead,
              precomputed: List[PrecomputedTransition], embeddings: torch.Tensor,
              config: PRISMConfig, device: torch.device) -> Dict[str, Any]:
    if not precomputed:
        raise ValueError("No transitions available for IQL training")
    state_idx = torch.tensor([p.state_key for p in precomputed], dtype=torch.long)
    sa_idx = torch.tensor([p.state_action_key for p in precomputed], dtype=torch.long)
    next_idx = torch.tensor([p.next_state_key for p in precomputed], dtype=torch.long)
    r_hat = torch.tensor([p.r_hat for p in precomputed], dtype=torch.float32)
    terminal = torch.tensor([p.terminal for p in precomputed], dtype=torch.float32)
    n = len(precomputed)

    embeddings = embeddings.to(device)
    value_head.to(device)
    q_head.to(device)
    q_target.to(device)
    q_target.load_state_dict(q_head.state_dict())
    for p in q_target.parameters():
        p.requires_grad = False

    params = list(value_head.parameters()) + list(q_head.parameters())
    optimizer = build_optimizer(params, config.iql_lr, config.iql_weight_decay, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.iql_warmup_ratio * config.iql_steps), config.iql_steps)
    generator = torch.Generator().manual_seed(config.seed)
    history: List[Dict[str, float]] = []

    for step in range(1, config.iql_steps + 1):
        batch = torch.randint(0, n, (min(config.iql_batch_size, n),), generator=generator)
        s_emb = embeddings[state_idx[batch].to(device)]
        sa_emb = embeddings[sa_idx[batch].to(device)]
        ns_emb = embeddings[next_idx[batch].to(device)]
        r = r_hat[batch].to(device)
        d = terminal[batch].to(device)

        v_s = value_head(s_emb)
        with torch.no_grad():
            q_bar = q_target(sa_emb)
        loss_v = expectile_loss(q_bar - v_s, config.iql_expectile_tau)

        q_sa = q_head(sa_emb)
        with torch.no_grad():
            target = r + config.iql_gamma * (1.0 - d) * value_head(ns_emb)
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
        if step % config.log_every == 0:
            history.append({"step": float(step), "loss_v": float(loss_v.item()),
                            "loss_q": float(loss_q.item())})
            logger.info(f"[B3][IQL] step={step}/{config.iql_steps} "
                        f"loss_v={loss_v.item():.4f} loss_q={loss_q.item():.4f}")
    return {"history": history}


@dataclass
class AWRSample:
    prompt: str
    action_text: str
    weight: float


def build_awr_samples(precomputed: List[PrecomputedTransition], transitions: List[StepTransition],
                      trajectories_by_id: Dict[str, Trajectory], embeddings: torch.Tensor,
                      value_head: ScalarHead, q_head: ScalarHead, policy: BasePolicy,
                      config: PRISMConfig, device: torch.device) -> List[AWRSample]:
    value_head.eval()
    q_head.eval()
    embeddings = embeddings.to(device)
    advantages: List[float] = []
    with torch.no_grad():
        for i in range(0, len(precomputed), config.iql_batch_size):
            chunk = precomputed[i:i + config.iql_batch_size]
            s_idx = torch.tensor([p.state_key for p in chunk], dtype=torch.long, device=device)
            sa_idx = torch.tensor([p.state_action_key for p in chunk], dtype=torch.long, device=device)
            advantages.extend(
                (q_head(embeddings[sa_idx]) - value_head(embeddings[s_idx])).cpu().numpy().tolist())

    beta = config.awr_beta
    cap = math.exp(beta * config.awr_delta_max)
    samples: List[AWRSample] = []
    for tr, advantage in zip(transitions, advantages):
        traj = trajectories_by_id[tr.trajectory_id]
        action = traj.action_at(tr.step_index)
        if not action:
            continue
        weight = min(math.exp(min(beta * advantage, beta * config.awr_delta_max)), cap)
        if advantage < 0:
            weight *= config.awr_negative_advantage_weight
        samples.append(AWRSample(
            policy.build_prompt(tr.benchmark, traj.prefix_text(tr.step_index)),
            action, float(weight)))
    return samples


def tokenize_awr_item(prompt: str, action: str, tokenizer,
                      config: PRISMConfig) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                           max_length=config.policy_max_length)["input_ids"]
    action_ids = tokenizer(action, add_special_tokens=False, truncation=True,
                           max_length=config.action_max_tokens)["input_ids"]
    if tokenizer.eos_token_id is not None:
        action_ids = action_ids + [tokenizer.eos_token_id]
    return prompt_ids + action_ids, [-100] * len(prompt_ids) + list(action_ids)


def collate_awr(batch: List[AWRSample], tokenizer,
                config: PRISMConfig) -> Dict[str, torch.Tensor]:
    input_ids_list, attention_list, labels_list, weights = [], [], [], []
    for item in batch:
        full_ids, labels = tokenize_awr_item(item.prompt, item.action_text, tokenizer, config)
        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        attention_list.append(torch.ones(len(full_ids), dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        weights.append(item.weight)
    return {
        "input_ids": pad_sequence(input_ids_list, batch_first=True,
                                  padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence(attention_list, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels_list, batch_first=True, padding_value=-100),
        "weights": torch.tensor(weights, dtype=torch.float32),
    }


def compute_per_sequence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    flat = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    flat = flat.view(shift_labels.size())
    valid = (shift_labels != -100).float()
    return (flat * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def run_weighted_lm_training(policy: BasePolicy, samples: List[AWRSample], adapter: str,
                             total_steps: int, lr: float, weight_decay: float,
                             warmup_ratio: float, batch_size: int, config: PRISMConfig,
                             stage: str) -> Dict[str, Any]:
    if not samples:
        logger.warning(f"[{stage}] no samples available; skipping")
        return {"history": []}
    policy.add_adapter(adapter)
    policy.enable_training_mode()
    model, tokenizer = policy.model, policy.tokenizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable, lr, weight_decay, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps)
    rng = random.Random(config.seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    cursor = 0
    history: List[Dict[str, float]] = []
    for step in range(1, total_steps + 1):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        batch_items = [samples[i] for i in order[cursor:cursor + batch_size]]
        cursor += batch_size
        batch = collate_awr(batch_items, tokenizer, config)
        outputs = model(input_ids=batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device))
        seq_losses = compute_per_sequence_loss(outputs.logits.float(),
                                               batch["labels"].to(model.device))
        weights = batch["weights"].to(model.device)
        loss = (weights * seq_losses).sum() / weights.sum().clamp(min=1e-8)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        scheduler.step()
        if step % config.log_every == 0:
            history.append({"step": float(step), "loss": float(loss.item())})
            logger.info(f"[{stage}] step={step}/{total_steps} loss={loss.item():.4f}")
        if config.save_every and step % config.save_every == 0:
            model.save_pretrained(str(Path(config.output_dir) / f"{adapter}_adapter_step{step}"),
                                  selected_adapters=[adapter])
    policy.enable_inference_mode()
    return {"history": history}


def train_sft_policy(policy: BasePolicy, buffer: List[Trajectory],
                     config: PRISMConfig) -> Dict[str, Any]:
    samples: List[AWRSample] = []
    for traj in buffer:
        if not traj.success:
            continue
        for t in range(traj.length):
            action = traj.action_at(t)
            if action:
                samples.append(AWRSample(
                    policy.build_prompt(traj.benchmark, traj.prefix_text(t)), action, 1.0))
    total_steps = max(1, config.sft_epochs * max(1, len(samples) // max(config.awr_batch_size, 1)))
    logger.info(f"[A][SFT] {len(samples)} samples over {config.sft_epochs} epoch(s) "
                f"= {total_steps} steps")
    return run_weighted_lm_training(policy, samples, "sft", total_steps, config.awr_lr,
                                    config.awr_weight_decay, config.awr_warmup_ratio,
                                    config.awr_batch_size, config, "A][SFT")


def train_awr_policy(policy: BasePolicy, samples: List[AWRSample],
                     config: PRISMConfig) -> Dict[str, Any]:
    epochs = config.awr_steps * config.awr_batch_size / max(len(samples), 1)
    logger.info(f"[B4][AWR] {len(samples)} step transitions, {config.awr_steps} updates at batch "
                f"{config.awr_batch_size} (~{epochs:.1f} epochs)")
    return run_weighted_lm_training(policy, samples, "awr", config.awr_steps, config.awr_lr,
                                    config.awr_weight_decay, config.awr_warmup_ratio,
                                    config.awr_batch_size, config, "B4][AWR")


def self_correcting_inference(policy: BasePolicy, prm: ProcessRewardModel, conf: ConfidenceHead,
                              env: BaseEnv, task_id: str, config: PRISMConfig,
                              device: torch.device, adapter: Optional[str],
                              tau_low: Optional[float] = None,
                              tau_abort: Optional[float] = None) -> Trajectory:
    tau_low = config.tau_low if tau_low is None else tau_low
    tau_abort = config.tau_abort if tau_abort is None else tau_abort
    horizon = config.horizon(env.benchmark)
    obs, goal = env.reset(task_id)
    traj = Trajectory(
        trajectory_id=f"eval_{env.benchmark}_{task_id}_{int(time.time() * 1e6)}",
        benchmark=env.benchmark,
        task_id=task_id,
        goal_text=goal,
    )
    traj.states.append(obs)
    tokens_before = policy.decoded_tokens
    last_score = 0.0
    with policy.using(adapter):
        for t in range(horizon):
            history = traj.history_text(t)
            prefix_text = traj.prefix_text(t)
            reflection: Optional[str] = None
            chosen_action: Optional[str] = None
            used_reflection = False
            for retry in range(config.r_max + 1):
                prompt = policy.build_prompt(env.benchmark, prefix_text, reflection=reflection)
                generations = policy.sample(prompt, k=config.k_candidates,
                                            temperature=config.decode_temperature)
                candidates = [policy.parse_action(g) for g in generations]
                scores = prm.score([(traj.goal_text, history, a) for a in candidates],
                                   device).cpu().numpy()
                best_action = candidates[int(np.argmax(scores))]
                chosen_action = best_action
                c_t = float(conf.confidence_calibrated(
                    [(traj.goal_text, history, best_action)], env.benchmark, device).cpu().numpy()[0])
                if c_t >= tau_low:
                    break
                if retry == config.r_max:
                    traj.aborted = c_t < tau_abort
                    break
                critique = policy.sample(
                    policy.build_reflection_prompt(env.benchmark, prefix_text, candidates,
                                                   scores.tolist()),
                    k=1, temperature=config.decode_temperature,
                    max_new_tokens=config.reflection_max_new_tokens)
                reflection = " ".join(critique[0].strip().split("\n")[:2]).strip()
                used_reflection = True
            if used_reflection:
                traj.reflection_steps += 1
            if traj.aborted or chosen_action is None:
                break
            traj.actions.append(chosen_action)
            next_obs, reward, terminal, info = env.step(chosen_action)
            traj.rewards.append(reward)
            traj.states.append(next_obs)
            traj.length += 1
            last_score = float(info.get("score", reward))
            if terminal:
                traj.success = bool(info.get(
                    "success", reward >= config.success_threshold(env.benchmark)))
                break
    traj.final_reward = last_score
    traj.decoded_tokens = policy.decoded_tokens - tokens_before
    return traj


def score_trajectory(traj: Trajectory, benchmark: str,
                     config: PRISMConfig) -> Tuple[bool, float]:
    if benchmark == "alfworld":
        return bool(traj.success), 1.0 if traj.success else 0.0
    if benchmark == "webshop":
        return traj.final_reward >= config.webshop_success_threshold, float(traj.final_reward)
    return traj.final_reward >= config.scienceworld_success_threshold, float(traj.final_reward)


def evaluate(policy: BasePolicy, prm: ProcessRewardModel, conf: ConfidenceHead,
             envs: Dict[str, BaseEnv], config: PRISMConfig, device: torch.device,
             adapter: Optional[str], max_tasks: Optional[int] = None) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for benchmark, env in envs.items():
        eval_tasks = env.list_eval_tasks()
        if max_tasks is not None:
            eval_tasks = eval_tasks[:max_tasks]
        metric = config.primary_metric(benchmark)
        per_seed: Dict[str, List[float]] = {
            "success_rate": [], "mean_reward": [], "steps_per_success": [],
            "tokens_per_task": [], "reflection_rate": [], "abort_rate": [],
        }
        per_task_success: Dict[str, List[float]] = {task: [] for task in eval_tasks}
        for seed in config.seeds_eval:
            set_seed(seed)
            successes, aborts, reflect_steps, total_steps = 0, 0, 0, 0
            scores, steps, tokens = [], [], []
            for task_id in eval_tasks:
                traj = self_correcting_inference(policy, prm, conf, env, task_id, config,
                                                 device, adapter)
                ok, score = score_trajectory(traj, benchmark, config)
                per_task_success[task_id].append(1.0 if ok else 0.0)
                if ok:
                    successes += 1
                    steps.append(traj.length)
                scores.append(score)
                tokens.append(traj.decoded_tokens)
                reflect_steps += traj.reflection_steps
                total_steps += max(traj.length, 1)
                aborts += int(traj.aborted)
            n = max(len(eval_tasks), 1)
            per_seed["success_rate"].append(successes / n)
            per_seed["mean_reward"].append(float(np.mean(scores)) if scores else 0.0)
            per_seed["steps_per_success"].append(float(np.mean(steps)) if steps else 0.0)
            per_seed["tokens_per_task"].append(float(np.mean(tokens)) if tokens else 0.0)
            per_seed["reflection_rate"].append(reflect_steps / max(total_steps, 1))
            per_seed["abort_rate"].append(aborts / n)
            logger.info(f"[C] {benchmark} seed={seed} {metric}={per_seed[metric][-1]:.4f}")
        mean = {k: float(np.mean(v)) if v else 0.0 for k, v in per_seed.items()}
        std = {k: float(np.std(v)) if v else 0.0 for k, v in per_seed.items()}
        results[benchmark] = {
            "primary_metric": metric,
            "n_eval_tasks": len(eval_tasks),
            "seeds": list(config.seeds_eval),
            "per_seed": per_seed,
            "mean": mean,
            "std": std,
            "per_task_success": per_task_success,
        }
        logger.info(f"[C] {benchmark}: {metric}={mean[metric]:.4f} +/- {std[metric]:.4f} "
                    f"steps/success={mean['steps_per_success']:.2f} "
                    f"tokens/task={mean['tokens_per_task']:.0f} "
                    f"reflection_rate={mean['reflection_rate']:.3f}")
    return results


def search_confidence_thresholds(policy: BasePolicy, prm: ProcessRewardModel,
                                 conf: ConfidenceHead, envs: Dict[str, BaseEnv],
                                 calibration_tasks: Dict[str, List[str]], config: PRISMConfig,
                                 device: torch.device, adapter: Optional[str]
                                 ) -> Tuple[float, float, List[Dict[str, float]]]:
    rng = random.Random(config.seed)
    tasks_by_benchmark = {
        benchmark: rng.sample(tasks, min(config.threshold_search_tasks, len(tasks)))
        for benchmark, tasks in calibration_tasks.items() if tasks
    }
    grid: List[Dict[str, float]] = []
    best_low, best_abort, best_value = config.tau_low, config.tau_abort, -1.0
    for tau_low in config.tau_low_grid:
        for tau_abort in config.tau_abort_grid:
            if tau_abort >= tau_low:
                continue
            set_seed(config.seed)
            outcomes: List[float] = []
            for benchmark, tasks in tasks_by_benchmark.items():
                for task_id in tasks:
                    traj = self_correcting_inference(policy, prm, conf, envs[benchmark], task_id,
                                                     config, device, adapter, tau_low, tau_abort)
                    ok, _score = score_trajectory(traj, benchmark, config)
                    outcomes.append(1.0 if ok else 0.0)
            value = float(np.mean(outcomes)) if outcomes else 0.0
            grid.append({"tau_low": tau_low, "tau_abort": tau_abort, "calibration_success": value})
            logger.info(f"[B2][GRID] tau_low={tau_low:.2f} tau_abort={tau_abort:.2f} "
                        f"calibration_success={value:.4f}")
            if value > best_value:
                best_low, best_abort, best_value = tau_low, tau_abort, value
    logger.info(f"[B2][GRID] selected tau_low={best_low:.2f} tau_abort={best_abort:.2f} "
                f"calibration_success={best_value:.4f}")
    return best_low, best_abort, grid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_tasks(envs: Dict[str, BaseEnv], config: PRISMConfig
                ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    rng = random.Random(config.seed)
    train_tasks: Dict[str, List[str]] = {}
    calib_tasks: Dict[str, List[str]] = {}
    for benchmark, env in envs.items():
        tasks = list(env.list_train_tasks())[: config.n_train_tasks(benchmark)]
        shuffled = list(tasks)
        rng.shuffle(shuffled)
        n_calib = min(config.n_calib_tasks(benchmark), max(len(shuffled) - 1, 0))
        calib_tasks[benchmark] = sorted(shuffled[:n_calib])
        train_tasks[benchmark] = sorted(shuffled[n_calib:])
        logger.info(f"[SPLIT] {benchmark}: {len(train_tasks[benchmark])} training tasks, "
                    f"{len(calib_tasks[benchmark])} calibration tasks")
    return train_tasks, calib_tasks


def cap_tasks(tasks: Dict[str, List[str]], limit: Optional[int]) -> Dict[str, List[str]]:
    if limit is None:
        return tasks
    return {benchmark: task_list[:limit] for benchmark, task_list in tasks.items()}


def save_json(payload: Any, path: Path) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def run_pipeline(config: PRISMConfig, args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(asdict(config), out_dir / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = {b: build_env(b, config) for b in config.benchmarks}

    train_tasks, calib_tasks = split_tasks(envs, config)
    train_tasks = cap_tasks(train_tasks, args.max_tasks_per_env)
    calib_tasks = cap_tasks(calib_tasks, args.max_calib_tasks_per_env)
    save_json({"train": train_tasks, "calibration": calib_tasks}, out_dir / "task_split.json")

    policy = BasePolicy(config)
    buffer_path = out_dir / "trajectory_buffer.json"
    calib_path = out_dir / "calibration_pool.json"

    if args.reuse_buffer and buffer_path.exists() and calib_path.exists():
        buffer = [Trajectory.from_dict(d) for d in load_json(buffer_path)]
        calibration_pool = [Trajectory.from_dict(d) for d in load_json(calib_path)]
        logger.info(f"[A] reloaded {len(buffer)} training and "
                    f"{len(calibration_pool)} calibration trajectories")
        if args.run_sft:
            train_sft_policy(policy, buffer, config)
    else:
        logger.info("[A] collecting the calibration pool with pi_0 before anything is trained")
        calibration_pool = filter_and_deduplicate(
            collect_trajectories(policy, envs, calib_tasks, config.calib_rollouts_per_task,
                                 config, adapter=None, tag="calib"), config)
        logger.info(f"[A] calibration pool: {len(calibration_pool)} trajectories")

        logger.info(f"[A] collecting K_collect={config.k_collect} pi_0 rollouts per task")
        buffer = filter_and_deduplicate(
            collect_trajectories(policy, envs, train_tasks, config.k_collect,
                                 config, adapter=None, tag="pi0"), config)
        logger.info(f"[A] pi_0 buffer: {len(buffer)} trajectories")

        if args.run_sft:
            logger.info("[A] one-epoch SFT on the successful pi_0 subset to obtain pi_1")
            train_sft_policy(policy, buffer, config)
        else:
            logger.warning("[A] --run_sft not set; pi_1 falls back to pi_0")

        sft_adapter = "sft" if "sft" in policy.adapters else None
        logger.info(f"[A] collecting K_collect/2={config.k_collect_finetuned} pi_1 rollouts per task")
        buffer.extend(collect_trajectories(policy, envs, train_tasks, config.k_collect_finetuned,
                                           config, adapter=sft_adapter, tag="pi1"))
        buffer = filter_and_deduplicate(buffer, config)
        save_json([t.to_dict() for t in buffer], buffer_path)
        save_json([t.to_dict() for t in calibration_pool], calib_path)

    sft_adapter = "sft" if "sft" in policy.adapters else None
    save_json(summarize_buffer(buffer), out_dir / "buffer_composition.json")
    logger.info(f"[A] buffer composition: {json.dumps(summarize_buffer(buffer))}")

    labels_path = Path(args.step_labels_path) if args.step_labels_path else out_dir / "step_labels.json"
    if labels_path.exists() and not args.recompute_step_labels:
        triples = load_json(labels_path)
        logger.info(f"[B1] reusing {len(triples)} cached step labels from {labels_path}")
    else:
        triples = monte_carlo_step_labels(buffer, policy, envs, config, sft_adapter)
        save_json(triples, labels_path)
        logger.info(f"[B1] wrote {len(triples)} step labels to {labels_path}")
    triples = corrupt_step_labels(triples, config.label_noise_eps, config.seed)

    logger.info("[B2] training the process reward model and the confidence head")
    backbone = EncoderBackbone(config)
    prm = ProcessRewardModel(backbone).to(device)
    conf = ConfidenceHead(backbone).to(device)
    prm_stats = train_prm(prm, triples, config, device)
    conf_stats = train_confidence_head(conf, buffer, config, device)
    calibration_report = calibrate_confidence(conf, calibration_pool, config, device)
    save_json({"prm": prm_stats, "confidence": conf_stats, "calibration": calibration_report},
              out_dir / "stage_b2_report.json")
    torch.save(prm.state_dict(), out_dir / "prm.pt")
    torch.save({"state_dict": conf.state_dict(), "platt": conf.platt},
               out_dir / "confidence_head.pt")

    logger.info("[B3] precomputing behavioral rewards and frozen encoder states")
    trajectories_by_id = {t.trajectory_id: t for t in buffer}
    transitions: List[StepTransition] = []
    for traj in buffer:
        transitions.extend(traj.transitions())
    precomputed, embeddings = precompute_iql_transitions(
        transitions, trajectories_by_id, prm, backbone, config, device)
    value_head = ScalarHead(backbone.hidden_size)
    q_head = ScalarHead(backbone.hidden_size)
    q_target = ScalarHead(backbone.hidden_size)
    iql_stats = train_iql(value_head, q_head, q_target, precomputed, embeddings, config, device)
    save_json(iql_stats, out_dir / "stage_b3_report.json")
    torch.save({"value_head": value_head.state_dict(), "q_head": q_head.state_dict()},
               out_dir / "iql_critics.pt")

    logger.info("[B4] extracting the policy with advantage-weighted regression")
    awr_samples = build_awr_samples(precomputed, transitions, trajectories_by_id, embeddings,
                                    value_head, q_head, policy, config, device)
    awr_stats = train_awr_policy(policy, awr_samples, config)
    save_json(awr_stats, out_dir / "stage_b4_report.json")
    eval_adapter = "awr" if "awr" in policy.adapters else None
    if eval_adapter is not None:
        policy.model.save_pretrained(str(out_dir / "policy_adapter"),
                                     selected_adapters=[eval_adapter])

    thresholds: Dict[str, Any] = {"tau_low": config.tau_low, "tau_abort": config.tau_abort}
    if args.search_thresholds:
        logger.info("[B2] grid-searching the confidence thresholds on the calibration pool")
        tau_low, tau_abort, grid = search_confidence_thresholds(
            policy, prm, conf, envs, calib_tasks, config, device, eval_adapter)
        config.tau_low, config.tau_abort = tau_low, tau_abort
        thresholds = {"tau_low": tau_low, "tau_abort": tau_abort, "grid": grid}
    save_json(thresholds, out_dir / "thresholds.json")

    logger.info("[C] evaluating with PRM re-scoring and confidence-gated self-correction")
    results = evaluate(policy, prm, conf, envs, config, device, eval_adapter,
                       max_tasks=args.max_eval_tasks)
    save_json(results, out_dir / "results.json")
    for env in envs.values():
        env.close()
    return results


def build_config(args: argparse.Namespace) -> PRISMConfig:
    config = PRISMConfig(
        base_model_name=args.base_model_name,
        encoder_model_name=args.encoder_model_name,
        benchmarks=tuple(args.benchmarks),
        seed=args.seed,
        output_dir=args.output_dir,
        n_step_labels=args.n_step_labels,
        m_continuations=args.m_continuations,
        label_noise_eps=args.label_noise_eps,
        iql_steps=args.iql_steps,
        prm_epochs=args.prm_epochs,
        conf_epochs=args.conf_epochs,
        awr_steps=args.awr_steps,
        awr_beta=args.awr_beta,
        iql_expectile_tau=args.iql_expectile_tau,
        alpha_reward_mix=args.alpha_reward_mix,
        lambda_mse=args.lambda_mse,
        k_candidates=args.k_candidates,
        tau_low=args.tau_low,
        tau_abort=args.tau_abort,
        r_max=args.r_max,
        scienceworld_success_threshold=args.scienceworld_success_threshold,
    )
    if args.seeds_eval:
        config.seeds_eval = tuple(args.seeds_eval)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM end-to-end pipeline")
    parser.add_argument("--output_dir", type=str, default="./prism_outputs")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--encoder_model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds_eval", type=int, nargs="+", default=None)
    parser.add_argument("--max_tasks_per_env", type=int, default=None)
    parser.add_argument("--max_calib_tasks_per_env", type=int, default=None)
    parser.add_argument("--max_eval_tasks", type=int, default=None)
    parser.add_argument("--run_sft", action="store_true")
    parser.add_argument("--reuse_buffer", action="store_true")
    parser.add_argument("--step_labels_path", type=str, default=None)
    parser.add_argument("--recompute_step_labels", action="store_true")
    parser.add_argument("--search_thresholds", action="store_true")
    parser.add_argument("--n_step_labels", type=int, default=60000)
    parser.add_argument("--m_continuations", type=int, default=8)
    parser.add_argument("--label_noise_eps", type=float, default=0.0)
    parser.add_argument("--iql_steps", type=int, default=50000)
    parser.add_argument("--prm_epochs", type=int, default=20)
    parser.add_argument("--conf_epochs", type=int, default=20)
    parser.add_argument("--awr_steps", type=int, default=50000)
    parser.add_argument("--awr_beta", type=float, default=3.0)
    parser.add_argument("--iql_expectile_tau", type=float, default=0.7)
    parser.add_argument("--alpha_reward_mix", type=float, default=0.7)
    parser.add_argument("--lambda_mse", type=float, default=0.5)
    parser.add_argument("--k_candidates", type=int, default=3)
    parser.add_argument("--tau_low", type=float, default=0.45)
    parser.add_argument("--tau_abort", type=float, default=0.20)
    parser.add_argument("--r_max", type=int, default=2)
    parser.add_argument("--scienceworld_success_threshold", type=float, default=90.0)
    args = parser.parse_args()
    run_pipeline(build_config(args), args)


if __name__ == "__main__":
    main()
