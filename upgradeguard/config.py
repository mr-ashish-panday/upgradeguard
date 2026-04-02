from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

MAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
REPLICATION_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 256
TRAIN_SAMPLES = 1000
EVAL_SAMPLES = 200
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
PARTIAL_UNFREEZE_LAYERS = 4
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 3
SEED = 42
OUTPUT_DIR = "./results"
DEVICE = "cuda"

MAX_INPUT_TOKENS = 1024
MAX_SEQUENCE_TOKENS = 1536
GENERATION_BATCH_SIZE = 4
KL_PROMPT_SAMPLES = 50
SAFETY_EVAL_HARMFUL_SAMPLES = 100
SAFETY_EVAL_JAILBREAK_SAMPLES = 50
SAFETY_EVAL_BENIGN_SAMPLES = 50
PROMPT_CONSISTENCY_VARIANTS = 5
SMOKE_TEST_PROMPTS = 10
EXTERNAL_HARMBENCH_SAMPLES = 320
EXTERNAL_XSTEST_SAMPLES = 450
EXTERNAL_STRONGREJECT_SAMPLES = 60
POSTHOC_RISK_THRESHOLD = 0.05

TASKS: Tuple[str, ...] = ("summarization", "code_gen", "translation")
UPDATE_METHODS: Tuple[str, ...] = ("full_ft", "lora", "qlora", "partial_unfreeze")
MODEL_ALIASES: Dict[str, str] = {
    "main": MAIN_MODEL,
    "qwen": MAIN_MODEL,
    "replication": REPLICATION_MODEL,
    "llama": REPLICATION_MODEL,
}

PILOT_CONDITIONS: Tuple[Tuple[str, str, str], ...] = (
    (MAIN_MODEL, "summarization", "full_ft"),
    (MAIN_MODEL, "summarization", "lora"),
    (MAIN_MODEL, "code_gen", "full_ft"),
    (MAIN_MODEL, "code_gen", "lora"),
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    source_label: str
    target_label: str
    utility_metric: str


TASK_SPECS: Dict[str, TaskSpec] = {
    "summarization": TaskSpec(
        name="summarization",
        instruction="Summarize the following article accurately and concisely.",
        source_label="Article",
        target_label="Summary",
        utility_metric="rougeL",
    ),
    "code_gen": TaskSpec(
        name="code_gen",
        instruction="Write Python code that satisfies the requested behavior.",
        source_label="Task",
        target_label="Code",
        utility_metric="pass@1",
    ),
    "translation": TaskSpec(
        name="translation",
        instruction="Translate the following English text into fluent French.",
        source_label="English",
        target_label="French",
        utility_metric="bleu",
    ),
}


def output_root(root: str | Path | None = None) -> Path:
    return Path(root or OUTPUT_DIR)


def cache_root(root: str | Path | None = None) -> Path:
    return output_root(root) / "cache"


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_")


def resolve_model_name(model_arg: str | None) -> str:
    if not model_arg:
        return MAIN_MODEL
    return MODEL_ALIASES.get(model_arg.lower(), model_arg)


def resolve_model_selection(model_arg: str | None) -> Sequence[str]:
    if not model_arg or model_arg.lower() == "all":
        return [MAIN_MODEL, REPLICATION_MODEL]
    return [resolve_model_name(model_arg)]


def resolve_task_selection(task_arg: str | None) -> Sequence[str]:
    if not task_arg or task_arg.lower() == "all":
        return list(TASKS)
    if task_arg not in TASKS:
        raise ValueError(f"Unknown task '{task_arg}'. Expected one of {TASKS}.")
    return [task_arg]


def resolve_method_selection(method_arg: str | None) -> Sequence[str]:
    if not method_arg or method_arg.lower() == "all":
        return list(UPDATE_METHODS)
    if method_arg not in UPDATE_METHODS:
        raise ValueError(f"Unknown method '{method_arg}'. Expected one of {UPDATE_METHODS}.")
    return [method_arg]
