#!/usr/bin/env python3
"""
Phase 3 : Model Building and Training
Adaptation de ConfiDx pour le diagnostic du cancer du sein — Lot 1

Fine-tuning LoRA de Llama-3.1-8B-Instruct sur les 4 sous-taches (diagnostic,
explication, reconnaissance d'incertitude, explication d'incertitude), avec
quantification 4-bit (QLoRA), ordonnancement round-robin et ponderation de
classe pour la Task 3.

STATUT : pipeline valide sur smoke test (40 exemples/tache, 30 steps,
loss 4.4567 -> 0.5476). Entrainement complet (SMOKE_TEST=False) non execute
au moment de la redaction, faute de quota GPU Colab suffisant (voir
rapport Phase 3, Section 8 - Limitations).

Reconstitue depuis le rapport technique Phase 3 (16 Aout 2026) pour
versionner le code source reellement execute lors du smoke test.
Destine a etre execute sur Google Colab (chemins /content/...).
"""

# ============================================================
# CELL 4 : Configuration globale
# ============================================================
import json, math, random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Chemins (environnement Google Colab)
DATA_DIR = Path("/content/data/processed")
OUTPUT_DIR = Path("/content/drive/MyDrive/ConfiDx/models/phase3_lora")
LOG_DIR = Path("/content/drive/MyDrive/ConfiDx/logs/phase3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Modele (Tier 0/1, methodologie Section 3.5.1)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
USE_4BIT = True

# Test rapide du pipeline avant lancement complet
SMOKE_TEST = True
SMOKE_TEST_N_PER_TASK = 40

# Hyperparametres LoRA (Section 3.5.2, fideles a la methodologie)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Hyperparametres d'entrainement
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 1024
LOGGING_STEPS = 10
SAVE_STEPS = 200
EXPECTED_MIN_STEPS_MARGIN = 0.9


# ============================================================
# CELL 5 : Chargement multi-taches et poids de classe Task 3
# ============================================================
TASK_NAMES = {1: "diagnosis", 2: "explanation",
              3: "uncertainty", 4: "uncertainty_explanation"}

def load_task_file(split: str, task_num: int):
    file_path = DATA_DIR / split / f"task{task_num}_{split}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier non trouve : {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for ex in data:
        ex["task_id"] = task_num
    return data

def compute_task3_class_weights(train_task3):
    """wc = N / (K * nc), section 3.5.3"""
    labels = [ex["output"].strip().lower() for ex in train_task3]
    counts = Counter(labels)
    N = len(labels)
    K = len(counts)
    weights = {cls: N / (K * n) for cls, n in counts.items()}
    print("Distribution Task 3 (train):", dict(counts))
    print("Poids de classe calcules :", weights)
    return weights

def build_multitask_split(split: str):
    tasks = {t: load_task_file(split, t) for t in range(1, 5)}
    if SMOKE_TEST:
        for t in tasks:
            random.shuffle(tasks[t])
            tasks[t] = tasks[t][:SMOKE_TEST_N_PER_TASK]
        print(f"[SMOKE TEST] {split}: {SMOKE_TEST_N_PER_TASK} exemples/tache")
    return tasks

train_tasks = build_multitask_split("train")
val_tasks = build_multitask_split("val")
TASK3_WEIGHTS = compute_task3_class_weights(train_tasks[3])


# ============================================================
# CELL 6 : Dataset multi-taches et RoundRobinSampler
# ============================================================
from torch.utils.data import Dataset, Sampler

class ConfiDxMultiTaskDataset(Dataset):
    def __init__(self, tasks_dict, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        self.task_indices = {t: [] for t in tasks_dict}

        for task_id, examples in tasks_dict.items():
            for ex in examples:
                global_idx = len(self.examples)
                self.task_indices[task_id].append(global_idx)

                if task_id == 3:
                    output = ex["output"].strip().lower()
                    weight = TASK3_WEIGHTS.get(output, 1.0)
                else:
                    weight = 1.0

                self.examples.append({
                    "instruction": ex["instruction"],
                    "input": ex["input"],
                    "output": ex["output"],
                    "task_id": task_id,
                    "loss_weight": weight,
                    "patient_id": ex.get("patient_id"),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class RoundRobinSampler(Sampler):
    """
    Cycle explicitement Task1 -> Task2 -> Task3 -> Task4 -> Task1 ...
    conformement a la section 3.5.3 ("round-robin scheduling").
    """
    def __init__(self, task_indices: dict, seed=42):
        self.task_indices = {t: list(idx) for t, idx in task_indices.items()}
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        pools = {t: idx[:] for t, idx in self.task_indices.items()}
        for t in pools:
            rng.shuffle(pools[t])
        cursors = {t: 0 for t in pools}
        max_len = max(len(p) for p in pools.values())
        task_cycle = sorted(pools.keys())

        order = []
        for _ in range(max_len):
            for t in task_cycle:
                pool = pools[t]
                if not pool:
                    continue
                if cursors[t] >= len(pool):
                    rng.shuffle(pool)
                    cursors[t] = 0
                order.append(pool[cursors[t]])
                cursors[t] += 1
        return iter(order)

    def __len__(self):
        return max(len(p) for p in self.task_indices.values()) * len(self.task_indices)


# ============================================================
# CELL 8 : Chargement du modele en 4-bit (QLoRA) + config LoRA
# ============================================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# CELL 9 : DataLoader et collate_fn (masquage du prompt)
# ============================================================
from functools import partial
from torch.utils.data import DataLoader

def collate_fn(batch, tokenizer):
    max_len = MAX_SEQ_LEN
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_mask, labels, task_ids, loss_weights = [], [], [], [], []

    for ex in batch:
        user_content = ex["instruction"] + "\n\n" + ex["input"]
        prompt_text = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(ex["output"], add_special_tokens=False)
        full_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]

        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]

        # Masquage du prompt : loss calculee uniquement sur la reponse
        labels_ex = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
        if len(labels_ex) > len(full_ids):
            labels_ex = labels_ex[:len(full_ids)]
        elif len(labels_ex) < len(full_ids):
            labels_ex = labels_ex + [-100] * (len(full_ids) - len(labels_ex))

        pad_n = max_len - len(full_ids)
        input_ids.append(full_ids + [pad_token_id] * pad_n)
        attention_mask.append([1] * len(full_ids) + [0] * pad_n)
        labels.append(labels_ex + [-100] * pad_n)
        task_ids.append(ex["task_id"])
        loss_weights.append(ex.get("loss_weight", 1.0))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "task_ids": torch.tensor(task_ids, dtype=torch.long),
        "loss_weights": torch.tensor(loss_weights, dtype=torch.float),
    }

train_dataset = ConfiDxMultiTaskDataset(train_tasks, tokenizer)
val_dataset = ConfiDxMultiTaskDataset(val_tasks, tokenizer)
train_sampler = RoundRobinSampler(train_dataset.task_indices)

global_batch_size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
train_size = len(train_dataset)

train_dataloader = DataLoader(
    train_dataset, batch_size=PER_DEVICE_BATCH_SIZE, sampler=train_sampler,
    collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=0,
)
val_dataloader = DataLoader(
    val_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False,
    collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=0,
)


# ============================================================
# CELL 10 : Calcul explicite de max_steps (piege critique, section 3.5.4)
# ============================================================
# max_steps = epochs * ceil(train_size / global_batch_size)
# JAMAIS laisse au defaut du Trainer -> risque documente de cap silencieux
# a 21 steps (voir methodologie, section 3.5.4 et Appendix A).
MAX_STEPS = math.ceil(train_size / global_batch_size) * NUM_EPOCHS

print(f"max_steps = {MAX_STEPS}")
print(f"Train size : {train_size}")
print(f"Global batch : {global_batch_size}")
print(f"Steps/epoch : {math.ceil(train_size / global_batch_size)}")
print(f"Epochs : {NUM_EPOCHS}")


# ============================================================
# CELL 11 : Optimiseur, scheduler, checkpointing
# ============================================================
import time
import bitsandbytes as bnb
from transformers import get_linear_schedule_with_warmup

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = CHECKPOINT_DIR / "training_state.json"

optimizer = bnb.optim.PagedAdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
)

num_warmup_steps = max(1, int(0.03 * MAX_STEPS))
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=MAX_STEPS
)

def save_checkpoint(step):
    ckpt_path = CHECKPOINT_DIR / f"step_{step}"
    model.save_pretrained(str(ckpt_path))
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_path / "scheduler.pt")
    with open(STATE_FILE, "w") as f:
        json.dump({"last_step": step, "checkpoint_path": str(ckpt_path)}, f)
    print(f"Checkpoint sauvegarde : {ckpt_path} (step {step})")

def load_latest_checkpoint():
    if not STATE_FILE.exists():
        return 0
    with open(STATE_FILE) as f:
        state = json.load(f)
    ckpt_path = Path(state["checkpoint_path"])
    if not ckpt_path.exists():
        return 0
    print(f"Reprise depuis {ckpt_path} (step {state['last_step']})")
    model.load_adapter(str(ckpt_path), adapter_name="default")
    optimizer.load_state_dict(torch.load(ckpt_path / "optimizer.pt"))
    scheduler.load_state_dict(torch.load(ckpt_path / "scheduler.pt"))
    return state["last_step"]

start_step = load_latest_checkpoint()


# ============================================================
# CELL 12 : Boucle d'entrainement manuelle (loss ponderee + round-robin)
# ============================================================
model.train()
global_step = start_step
train_iter = iter(train_dataloader)
log_history = []
t0 = time.time()

while global_step < MAX_STEPS:
    micro_losses = []
    optimizer.zero_grad()

    for _ in range(GRAD_ACCUM_STEPS):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss_weights = batch.pop("loss_weights")
        batch.pop("task_ids", None)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"])
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size())

            valid_mask = (shift_labels != -100).float()
            per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / \
                valid_mask.sum(dim=1).clamp(min=1)

            # Ponderation de classe : wc = N/(K*nc), UNIQUEMENT Task 3
            loss = (per_example_loss * loss_weights).mean() / GRAD_ACCUM_STEPS

        loss.backward()
        micro_losses.append(loss.item() * GRAD_ACCUM_STEPS)

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    scheduler.step()

    global_step += 1
    step_loss = sum(micro_losses) / len(micro_losses)
    log_history.append({"step": global_step, "loss": step_loss})

    if global_step % LOGGING_STEPS == 0 or global_step == 1:
        elapsed = time.time() - t0
        print(f"Step {global_step}/{MAX_STEPS} | loss={step_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

    if global_step % SAVE_STEPS == 0:
        save_checkpoint(global_step)

save_checkpoint(global_step)
model.save_pretrained(str(OUTPUT_DIR / "final_adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final_adapter"))


# ============================================================
# CELL 12bis : Assertion post-entrainement (Appendix A)
# ============================================================
def assert_training_completed(log_history, expected_steps, margin=0.9):
    """
    Leve une erreur si l'entrainement s'est arrete prematurement
    (piege documente : max_steps par defaut du Trainer HuggingFace = 21).
    """
    if not log_history:
        raise AssertionError("Aucun log trouve - entrainement invalide.")
    last_step = log_history[-1]["step"]
    expected_min = int(expected_steps * margin)
    assert last_step >= expected_min, (
        f"Training stopped early at step {last_step}, expected >= {expected_min}."
    )
    print(f"OK - {last_step} steps realises (attendu >= {expected_min})")

assert_training_completed(log_history, MAX_STEPS)

# ============================================================
# RESULTATS DU SMOKE TEST (documentes, rapport Phase 3 Section 6.1)
# ============================================================
# Step 1  | loss=4.4567 | lr=2.00e-04 |   66.0s
# Step 10 | loss=1.4850 | lr=1.38e-04 |  640.2s
# Step 20 | loss=0.7629 | lr=6.90e-05 | 1278.9s
# Step 30 | loss=0.5476 | lr=0.00     | 1917.2s
#
# STATUT : smoke test valide (30/30 steps). Entrainement complet
# (SMOKE_TEST=False, 1887 steps estimes, ~33.5h) non execute au moment
# de la redaction du rapport, faute de quota GPU Colab suffisant.
