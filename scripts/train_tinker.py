"""LoRA SFT of the SecondRead classifier on Tinker.

usage:
  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/train_tinker.py --dry-run          # no network, no key
  HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/train_tinker.py --log-path runs/sft_v1 \
      [--data data/sft_v1.jsonl] [--epochs 1] [--max-steps 10] [--dev-split data/splits/en-ru.dev.jsonl]

A --log-path that already holds checkpoints is resumed, so give smoke tests (G3/G4,
--max-steps) their own --log-path.

Adapted from tinker_cookbook/supervised/train.py (tinker-cookbook 0.5.7): same pipelined
loop (forward_backward + optim_step of step N+1 are submitted before step N's results
are awaited), cookbook CheckpointManager / checkpoints.jsonl / metrics.jsonl, and resume
from the last checkpoint in --log-path. Differences, all deliberate:

  * The LoRA client is MLP-only: rank=16, train_mlp=True, train_attn=False,
    train_unembed=False. Attention LoRA breaks convert_lora_to_gguf on Qwen3.5 and
    unembed LoRA with tied embeddings gives a wrongly shaped tensor
    (docs/prep/01-checklist.md section 1, rows 3-4). Resume re-creates the client with
    the same flags instead of trusting checkpoint metadata, whose missing flags the SDK
    would default to True.
  * Checkpoints are taken on the submit side, right after optim_step N is queued and
    before step N+1 is. Tinker runs one client's requests in submission order, so the
    checkpoint holds exactly N+1 updates and resume restarts at step N+1. (The cookbook
    saves after N+1 is already queued but records batch N, so a resume repeats batches.)
  * Training rows are SecondRead conversations from scripts/build_sft.py. Every datum of
    the first batch (and of --dry-run-batches batches in --dry-run) is checked: the input
    starts with the exact inference prompt, nothing was truncated, and loss weight sits
    only on <label><|im_end|>.
  * Optional dev scoring during training: exact p(Major)/p(OK) on a fixed dev sample via
    forward() on both label endings, reported as dev/auroc, dev/pr_auc, dev/label_nll.

Learning rate defaults to tinker_cookbook.hyperparam_utils.get_lr(BASE_MODEL)
(4.905e-4 for Qwen/Qwen3.5-4B), with the cookbook's linear decay.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.renderers import Renderer, TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

from secondread import metrics
from secondread.prompt import BASE_MODEL, LABEL_IDS, LABELS, RENDERER, messages, renderer

logger = logging.getLogger("train_tinker")

TRAIN_PRICE_PER_M = 0.737  # USD / 1M training tokens for Qwen/Qwen3.5-4B (docs/prep/03 section 3)
LORA = dict(train_mlp=True, train_attn=False, train_unembed=False)  # never change: see docstring


# --------------------------------------------------------------------------- data
def to_datum(conv: list[dict], ren: Renderer, max_length: int) -> tinker.Datum:
    # One assistant turn per conversation, so LAST_ASSISTANT_MESSAGE == ALL_ASSISTANT_MESSAGES
    # (and avoids the cookbook's extension-property warning). reduction="mean" (cookbook
    # default) makes each example's weights sum to 1.
    return conversation_to_datum(conv, ren, max_length, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)


def check_datum(datum: tinker.Datum, conv: list[dict], ren: Renderer, im_end: int) -> None:
    """Raise unless the datum is <inference prompt> -> <label><|im_end|> with loss on those 2 tokens only."""
    label = conv[-1]["content"]
    prompt = ren.build_generation_prompt(conv[:-1]).to_ints()
    inp = datum.model_input.to_ints()
    tgt = datum.loss_fn_inputs["target_tokens"].data
    w = datum.loss_fn_inputs["weights"].data
    # Target i is the token after input i, so the label (full-sequence index len(prompt)) is target len(prompt)-1.
    want = [len(prompt) - 1, len(prompt)]
    on = [i for i, x in enumerate(w) if x > 0]
    problems = []
    if inp[: len(prompt)] != prompt:
        problems.append("input does not start with the inference prompt")
    if on != want:
        problems.append(f"loss weight on positions {on}, expected {want}")
    elif [tgt[i] for i in on] != [LABEL_IDS[label], im_end]:
        problems.append(f"weighted targets {[tgt[i] for i in on]} != [{LABEL_IDS[label]}, {im_end}]")
    if len(tgt) != len(inp) or len(inp) != len(prompt) + 1:
        problems.append(f"lengths input={len(inp)} target={len(tgt)} prompt={len(prompt)} (truncated?)")
    if abs(sum(w) - 1.0) > 1e-5:
        problems.append(f"weights sum to {sum(w)}, expected 1 (mean reduction)")
    if problems:
        raise ValueError(f"bad datum for {conv[0]['content'][:60]!r}: " + "; ".join(problems))


class ConversationDataset(SupervisedDataset):
    """build_sft.py conversations, reshuffled per epoch with a seed derived from (seed, epoch),
    so a resumed run sees exactly the batches the interrupted run would have seen.
    Like the cookbook, the last partial batch of each epoch is dropped."""

    def __init__(self, rows: list[dict], batch_size: int, ren: Renderer, max_length: int, seed: int):
        self.rows, self.batch_size, self.ren, self.max_length, self.seed = rows, batch_size, ren, max_length, seed
        self.set_epoch(0)

    def set_epoch(self, seed: int = 0) -> None:
        self.order = list(range(len(self.rows)))
        random.Random(f"{self.seed}:{seed}").shuffle(self.order)

    def batch_rows(self, index: int) -> list[dict]:
        return [self.rows[j] for j in self.order[index * self.batch_size:(index + 1) * self.batch_size]]

    def get_batch(self, index: int) -> list[tinker.Datum]:
        return [to_datum(r["messages"], self.ren, self.max_length) for r in self.batch_rows(index)]

    def __len__(self) -> int:
        return len(self.rows) // self.batch_size


def load_rows(path: str) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    for r in rows:
        if r["messages"][-1]["content"] not in LABELS:
            raise ValueError(f"{r.get('id')}: assistant turn must be one of {LABELS}")
    return rows, hashlib.sha256(raw).hexdigest()[:16]


# --------------------------------------------------------------------------- dev scoring
class DevScorer:
    """Exact p(Major), p(OK) for a fixed dev sample using the *training* client.

    Each row is sent twice, once ending in each label; forward() returns the logprob of
    every target token, and the one at the label position is log p(label | prompt). No
    sampler export is needed, and the numbers are the same quantity eval_tinker.py
    and probe_local.py report (risk = p(Major) / (p(Major) + p(OK)))."""

    def __init__(self, rows: list[dict], ren: Renderer, max_length: int, im_end: int):
        self.rows, self.data, self.pos = rows, [], []
        for r in rows:
            for label in LABELS:  # data[2k] ends in Major, data[2k+1] in OK
                conv = messages(r["src"], r["mt"], r["lp"], label)
                d = to_datum(conv, ren, max_length)
                check_datum(d, conv, ren, im_end)
                self.data.append(d)
            # target index of the label token = prompt length - 1 (see check_datum)
            self.pos.append(len(ren.build_generation_prompt(messages(r["src"], r["mt"], r["lp"])).to_ints()) - 1)

    @property
    def n_tokens(self) -> int:
        return sum(d.model_input.length for d in self.data)

    async def __call__(self, tc: tinker.TrainingClient) -> dict[str, float]:
        res = await (await tc.forward_async(self.data, loss_fn="cross_entropy")).result_async()
        lps = [o["logprobs"].data for o in res.loss_fn_outputs]
        scored, nll = [], 0.0
        for k, (r, p) in enumerate(zip(self.rows, self.pos)):
            lp = dict(zip(LABELS, (lps[2 * k][p], lps[2 * k + 1][p])))
            log_z = max(lp.values()) + math.log(sum(math.exp(v - max(lp.values())) for v in lp.values()))
            nll -= lp[r["label"]] - log_z
            scored.append({"id": r["id"], "label": r["label"], "doc": r.get("doc"), "lp": r["lp"],
                           "risk": 1.0 / (1.0 + math.exp(lp["OK"] - lp["Major"]))})
        y, s = metrics.to_arrays(scored)
        return {"dev/auroc": metrics.auroc(y, s), "dev/pr_auc": metrics.pr_auc(y, s),
                "dev/label_nll": nll / len(scored), "dev/mean_risk": float(s.mean())}


def load_dev(path: str | None, n: int, seed: int) -> list[dict]:
    if not path:
        return []
    rows = metrics.load_scored(path)  # any jsonl
    random.Random(seed).shuffle(rows)
    return rows[:n]


# --------------------------------------------------------------------------- training
@dataclass
class Submitted:
    fwd_bwd: object
    optim: object
    step: int
    data: list
    metrics: dict
    t_submit: float


def planned_steps(a: argparse.Namespace, n_batches: int) -> int:
    return n_batches * a.epochs if a.max_steps is None else min(n_batches * a.epochs, a.max_steps)


def check_batch(ds: ConversationDataset, step: int) -> list[tinker.Datum]:
    """Build and check the batch of global step `step` (offline; raises on a bad datum)."""
    epoch, batch = divmod(step, len(ds))
    ds.set_epoch(epoch)
    im_end = ds.ren.tokenizer.convert_tokens_to_ids("<|im_end|>")
    data = ds.get_batch(batch)
    for d, r in zip(data, ds.batch_rows(batch)):
        check_datum(d, r["messages"], ds.ren, im_end)
    return data


def loop_state(step: int, n_batches: int, elapsed_tokens: int, data_sha: str) -> dict:
    """State recorded with a checkpoint taken after `step` steps: (epoch, batch) of the NEXT step."""
    return {"step": step, "epoch": step // n_batches, "batch": step % n_batches,
            "elapsed_tokens": elapsed_tokens, "data_sha": data_sha, "n_batches": n_batches}


def label_logprob_metrics(data: list[tinker.Datum], logprobs: list) -> dict[str, float]:
    """NLL of the label token alone (the im_end token is trivially predictable), split by class."""
    out: dict[str, list[float]] = {"label_nll": [], "label_nll_major": [], "label_nll_ok": []}
    for d, lp in zip(data, logprobs):
        w, t, lp = d.loss_fn_inputs["weights"].data, d.loss_fn_inputs["target_tokens"].data, lp.data
        i = next(i for i, x in enumerate(w) if x > 0)
        out["label_nll"].append(-lp[i])
        out["label_nll_major" if t[i] == LABEL_IDS["Major"] else "label_nll_ok"].append(-lp[i])
    return {k: sum(v) / len(v) for k, v in out.items() if v}


async def train(a: argparse.Namespace, ds: ConversationDataset, data_sha: str, dev: DevScorer | None) -> None:
    log_path = Path(a.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    n_batches = len(ds)
    total_steps = planned_steps(a, n_batches)

    resume = checkpoint_utils.get_last_checkpoint(str(log_path))  # needs a state_path
    start_step, elapsed_tokens = 0, 0
    if resume:
        if resume.get("data_sha", data_sha) != data_sha or resume.get("n_batches", n_batches) != n_batches:
            raise SystemExit(f"{log_path} holds a run on different data ({resume.get('data_sha')}, "
                             f"{resume.get('n_batches')} batches); use a new --log-path")
        step = resume.get("step")
        start_step = int(step) if step is not None else (resume.epoch or 0) * n_batches + (resume.batch or 0)
        elapsed_tokens = int(resume.get("elapsed_tokens", 0))
        if start_step >= total_steps:
            logger.info("Training already complete (step %d of %d); nothing to do", start_step, total_steps)
            return
    first = check_batch(ds, start_step)  # fail fast, before any Tinker request

    ml_logger = ml_log.setup_logging(log_dir=str(log_path), config=vars(a), do_configure_logging_module=True)
    service = tinker.ServiceClient(base_url=a.base_url, user_metadata=recipe_user_metadata("secondread_sft"))
    user_metadata: dict[str, str] = {"data_sha": data_sha}
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, a.renderer)
    tc = await service.create_lora_training_client_async(
        base_model=a.model, rank=a.rank, seed=a.seed, user_metadata=user_metadata, **LORA)
    if resume:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(service, resume.state_path, a.renderer)
        info = await service.create_rest_client().get_weights_info_by_tinker_path(resume.state_path).result_async()
        if info.lora_rank != a.rank or info.train_attn or info.train_unembed:
            raise SystemExit(f"checkpoint LoRA config {info} does not match rank={a.rank} {LORA}")
        await (await tc.load_state_with_optimizer_async(resume.state_path)).result_async()
        logger.info("Resumed from %s at step %d", resume.state_path, start_step)
    logger.info("Training client %s; %d steps (%d batches x %d epochs, max_steps=%s)",
                tc.get_console_url(), total_steps, n_batches, a.epochs, a.max_steps)

    ckpt = checkpoint_utils.CheckpointManager(
        training_client=tc, service_client=service, log_path=str(log_path),
        save_every=a.save_every, ttl_seconds=a.ttl_seconds, store=ml_logger.store)
    ckpt.restore_last_saved_tokens(elapsed_tokens)
    logger.info("first example:\n%s", colorize_example(first[0], ds.ren.tokenizer))

    async def submit(step: int) -> Submitted:
        epoch, batch = divmod(step, n_batches)
        if batch == 0:
            ds.set_epoch(epoch)  # check_batch() already set the epoch of start_step
        data = first if step == start_step else ds.get_batch(batch)
        lr = a.lr * compute_schedule_lr_multiplier(lr_schedule=a.lr_schedule, step=step, total_steps=total_steps)
        m: dict = {"epoch": epoch, "progress": (step + 1) / total_steps, "learning_rate": lr}
        if dev is not None and a.eval_every > 0 and step % a.eval_every == 0:
            m.update(await dev(tc))  # pre-step weights, as in the cookbook
        fwd_bwd = await tc.forward_backward_async(data, loss_fn="cross_entropy")
        optim = await tc.optim_step_async(tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
        return Submitted(fwd_bwd, optim, step, data, m, time.perf_counter())

    async def finish(s: Submitted) -> None:
        fb = await s.fwd_bwd.result_async()
        opt = await s.optim.result_async()
        logprobs = [o["logprobs"] for o in fb.loss_fn_outputs]
        s.metrics.update(opt.metrics or {})
        s.metrics.update(
            train_mean_nll=compute_mean_nll(logprobs, [d.loss_fn_inputs["weights"] for d in s.data]),
            num_sequences=len(s.data), num_tokens=sum(d.model_input.length for d in s.data),
            time_step=time.perf_counter() - s.t_submit, **label_logprob_metrics(s.data, logprobs))
        ml_logger.log_metrics(metrics=s.metrics, step=s.step)

    pending: deque[Submitted] = deque()
    for step in range(start_step, total_steps):
        pending.append(await submit(step))
        elapsed_tokens += sum(d.model_input.length for d in pending[-1].data)
        if step + 1 < total_steps:  # the final checkpoint below covers the last step
            # Queued after optim_step `step` and before step+1 -> holds exactly step+1 updates.
            await ckpt.maybe_save_async(step=step + 1, elapsed_tokens=elapsed_tokens,
                                        loop_state=loop_state(step + 1, n_batches, elapsed_tokens, data_sha))
        if len(pending) > a.submit_ahead:
            await finish(pending.popleft())
    while pending:
        await finish(pending.popleft())

    if dev is not None and a.eval_every > 0:
        final = await dev(tc)
        ml_logger.log_metrics(metrics=final, step=total_steps)
    paths = await ckpt.save_final_async(loop_state=loop_state(total_steps, n_batches, elapsed_tokens, data_sha))
    ml_logger.close()
    logger.info("final checkpoint: %s", paths)
    print(f"\nsampler weights: {paths.get('sampler_path')}\nnext: HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python "
          f"scripts/eval_tinker.py data/splits/en-ru.dev.jsonl --model-path {paths.get('sampler_path')} "
          f"--out results/<name>_enru_dev.jsonl")


# --------------------------------------------------------------------------- dry run
def dry_run(a: argparse.Namespace, ds: ConversationDataset, data_sha: str, dev: DevScorer | None) -> None:
    """Everything up to the first network call: batches, datum checks, schedule, cost."""
    tokenizer = ds.ren.tokenizer
    n_batches = len(ds)
    total_steps = planned_steps(a, n_batches)
    print(f"data: {a.data} sha={data_sha} rows={len(ds.rows)} -> {n_batches} batches of {a.batch_size} "
          f"({len(ds.rows) - n_batches * a.batch_size} rows dropped per epoch as a partial batch)")
    print(f"model={a.model} renderer={a.renderer} LoRA rank={a.rank} {LORA} seed={a.seed}")
    print(f"lr={a.lr:.4g} ({a.lr_schedule}), epochs={a.epochs}, max_steps={a.max_steps} -> {total_steps} steps, "
          f"max_length={a.max_length}, submit_ahead={a.submit_ahead}")
    sched = [0, total_steps // 4, total_steps // 2, 3 * total_steps // 4, total_steps - 1]
    print("lr at steps " + ", ".join(
        f"{s}: {a.lr * compute_schedule_lr_multiplier(a.lr_schedule, s, total_steps):.3g}" for s in sched))
    saves = [s for s in range(a.save_every, total_steps, a.save_every)] if a.save_every > 0 else []
    print(f"checkpoints (state + sampler weights) after steps {saves} + final at {total_steps}; "
          f"periodic ttl={a.ttl_seconds}s")
    resume = checkpoint_utils.get_last_checkpoint(a.log_path) if Path(a.log_path).exists() else None
    print(f"resume: {'from ' + resume.state_path + ' at step ' + str(resume.get('step')) if resume else 'none, fresh run'} "
          f"({a.log_path})")

    n_checked, tok_checked, n_major = 0, 0, 0
    for b in range(min(a.dry_run_batches, n_batches)):
        data, rows = check_batch(ds, b), ds.batch_rows(b)
        n_checked += len(data)
        tok_checked += sum(d.model_input.length for d in data)
        n_major += sum(r["label"] == "Major" for r in rows)
        print(f"batch {b}: {len(data)} datums OK, {sum(d.model_input.length for d in data)} input tokens, "
              f"{sum(r['label'] == 'Major' for r in rows)} Major, max len {max(d.model_input.length for d in data)}")
    d0 = ds.get_batch(0)[0]
    w, t = d0.loss_fn_inputs["weights"].data, d0.loss_fn_inputs["target_tokens"].data
    print("example 0 weighted targets:", [(t[i], tokenizer.decode([t[i]]), round(w[i], 3)) for i in range(len(w)) if w[i] > 0])
    print("example 0 prompt tail:", repr(tokenizer.decode(d0.model_input.to_ints()[-12:])))
    print(f"checked {n_checked} datums: loss weight only on <label><|im_end|>, prefix == inference prompt, "
          f"no truncation ({n_major} Major)")

    # Billing basis: input tokens of each datum (sequence length - 1), as cookbook logs num_tokens.
    used = [ds.rows[j] for j in ds.order[: n_batches * a.batch_size]]
    if all(r.get("n_tokens") for r in used):
        per_epoch = sum(r["n_tokens"] - 1 for r in used)
    else:  # file without n_tokens: extrapolate from the checked batches
        per_epoch = tok_checked * n_batches // max(1, min(a.dry_run_batches, n_batches))
    steps_frac = total_steps / n_batches
    cost = per_epoch * steps_frac * a.price / 1e6
    weighted = 2 * a.batch_size * total_steps
    print(f"tokens: {per_epoch:,}/epoch -> {int(per_epoch * steps_frac):,} for {total_steps} steps; "
          f"est. ${cost:.2f} if all tokens are billed, ${weighted * a.price / 1e6:.3f} if only weighted "
          f"({weighted:,}) are (@ ${a.price}/M)")
    if dev is not None:
        n_evals = (len(range(0, total_steps, a.eval_every)) + 1) if a.eval_every > 0 else 0
        print(f"dev scorer: {len(dev.rows)} rows x 2 labels, {dev.n_tokens:,} tokens per eval, {n_evals} evals "
              f"-> ~{dev.n_tokens * n_evals:,} forward tokens (~${dev.n_tokens * n_evals * a.price / 1e6:.2f} "
              f"if billed like training)")
    print("dry run OK: no network calls were made")


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT on Tinker (MLP-only).")
    ap.add_argument("--data", default="data/sft_v1.jsonl")
    ap.add_argument("--log-path", default="runs/sft_v1", help="metrics.jsonl, checkpoints.jsonl; resume source")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--renderer", default=RENDERER)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default: hyperparam_utils.get_lr(model)")
    ap.add_argument("--lr-schedule", default="linear", choices=["linear", "cosine", "constant"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None, help="cap, e.g. 10 for the G3 smoke test")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--ttl-seconds", type=int, default=7 * 24 * 3600, help="periodic checkpoints; final never expires")
    ap.add_argument("--submit-ahead", type=int, default=1, help="pipeline depth (0 = no pipelining)")
    ap.add_argument("--seed", type=int, default=0, help="LoRA init + per-epoch shuffle")
    ap.add_argument("--dev-split", default=None, help="raw split jsonl to score during training, e.g. en-ru.dev")
    ap.add_argument("--dev-n", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--price", type=float, default=TRAIN_PRICE_PER_M)
    ap.add_argument("--dry-run", action="store_true", help="build and check batches; no network, no API key")
    ap.add_argument("--dry-run-batches", type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.lr is None:
        a.lr = get_lr(a.model)
    if a.renderer != RENDERER or a.model != BASE_MODEL:
        logger.warning("model/renderer differ from secondread.prompt; inference will not match training")

    rows, data_sha = load_rows(a.data)
    ren = renderer()
    ds = ConversationDataset(rows, a.batch_size, ren, a.max_length, a.seed)
    if len(ds) == 0:
        raise SystemExit("fewer rows than one batch")
    im_end = ren.tokenizer.convert_tokens_to_ids("<|im_end|>")
    dev_rows = load_dev(a.dev_split, a.dev_n, a.seed)
    dev = DevScorer(dev_rows, ren, a.max_length, im_end) if dev_rows else None

    if a.dry_run:
        dry_run(a, ds, data_sha, dev)
    else:
        asyncio.run(train(a, ds, data_sha, dev))


if __name__ == "__main__":
    main()
