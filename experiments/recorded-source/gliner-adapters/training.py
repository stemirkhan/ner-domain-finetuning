"""Strict, auditable wrapper of GLiNER2 2.1.0's official ExtractorTrainer.

Never rents hardware or mutates the corpus. Checkpoints contain weights, not a
complete optimizer/RNG resume state. All output directories are write-once.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from common import ROOT, load_json, write_json, load_model, verify_data, seed_all


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StrictCollator:
    """Use all six fixed queries, verify the actual gold targets every batch."""

    def __init__(self, processor, schema, max_encoded_tokens=1024):
        self.processor = processor
        self.schema = schema
        self.max_encoded_tokens = max_encoded_tokens

    def __call__(self, records):
        labels = list(self.schema["names"].values())
        for text, output in records:
            if list(output["entities"]) != labels:
                raise ValueError("Expected all six entity queries in their fixed order")
            if output.get("entity_descriptions") != self.schema["descriptions"]:
                raise ValueError("Entity descriptions differ from the frozen schema")
            for values in output["entities"].values():
                for value in values:
                    if text[value["start"]:value["end"]] != value["text"]:
                        raise ValueError("Invalid explicit character span")
        # build_targets=True supplies supervision while inference-mode processing
        # disables all schema sampling. It does not put the neural model in eval.
        batch = self.processor.collate_fn_inference(
            records, architecture="boundary", max_len=None, error_policy="raise",
            build_targets=True, max_gold_per_query=64, on_capacity_exceeded="raise",
        )
        if len(batch) != len(records):
            raise ValueError("Processor dropped records")
        if int(batch.input_ids.shape[-1]) > self.max_encoded_tokens:
            raise ValueError("Encoded input exceeds cap; truncation is forbidden")
        for index, (_, output) in enumerate(records):
            expected = {(label, x["start"], x["end"])
                        for label, values in output["entities"].items() for x in values}
            actual = set()
            query_labels = []
            for qi, query in enumerate(batch.query_layouts[index].queries):
                query_labels.append(query.role_name)
                pairs = batch.targets.mention_pairs[index, qi]
                mask = batch.targets.mention_mask[index, qi]
                for a, b in pairs[mask].tolist():
                    actual.add((query.role_name, batch.start_mappings[index][a],
                                batch.end_mappings[index][b - 1]))
            if query_labels != labels or actual != expected:
                raise ValueError(f"Training target roundtrip failed for batch record {index}")
        return batch


def choose_pilot_indices(rows, lengths, count=64):
    """A deterministic train-only probe with long, negative, mixed and all types."""
    if not rows:
        raise ValueError("Empty train dataset")
    count = min(count, len(rows))
    chosen = []

    def add(index):
        if index not in chosen and len(chosen) < count:
            chosen.append(index)

    # The first is deliberately the longest record: resource pilots must not
    # size hardware using only short examples.
    add(max(range(len(rows)), key=lambda i: lengths[rows[i]["id"]]))
    negative = next((i for i, row in enumerate(rows) if not row["spans"]), None)
    if negative is not None:
        add(negative)
    mixed = next((i for i, row in enumerate(rows)
                  if len({s["label"] for s in row["spans"]}) > 1), None)
    if mixed is not None:
        add(mixed)
    labels = sorted({s["label"] for row in rows for s in row["spans"]})
    for label in labels:
        add(next(i for i, row in enumerate(rows) if any(s["label"] == label for s in row["spans"])))
    for index in sorted(range(len(rows)), key=lambda i: hashlib.sha256(rows[i]["id"].encode()).hexdigest()):
        add(index)
        if len(chosen) == count:
            break
    return chosen


def make_trainer_class():
    import torch
    from torch.utils.data import DataLoader
    from gliner2.training.trainer import ExtractorTrainer
    from gliner2.training.sampler import LengthGroupedSampler

    class RentalTrainer(ExtractorTrainer):
        def __init__(self, model, config, *, requested_device, schema, max_encoded_tokens, variant="lora"):
            self.variant = variant
            self.requested_device = requested_device
            self.fixed_schema = schema
            self.max_encoded_tokens = max_encoded_tokens
            self.saved_checkpoints = []
            self.checkpoint_seconds = 0.0
            self.processed_examples = 0
            self.successful_microbatches = 0
            super().__init__(model, config)

        def _setup_device(self):
            if self.requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
            self.device = torch.device(self.requested_device)
            self.is_distributed = False
            self.model.to(self.device)
            if self.variant != "qlora":
                self.model.float()

        def _setup_lora(self):
            if not self.config.use_lora:
                return
            from peft import LoraConfig, get_peft_model
            from gliner2.training.lora import _resolve_targets, _cast_lora_dtype, _PeftLoraLayer
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            targets = _resolve_targets(self.model, self.config.lora_target_modules)
            if len(targets) != 24:
                raise ValueError(f"Expected 24 encoder query/value linear modules, got {len(targets)}")
            config = LoraConfig(r=self.config.lora_r, lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout, target_modules=targets, bias="none",
                use_dora=self.variant == "dora", use_rslora=self.variant == "rslora",
                base_model_name_or_path=str(getattr(self.model, "name_or_path", "")))
            self.model = get_peft_model(self.model, config)
            if self.variant != "qlora":
                _cast_lora_dtype(self.model)
            else:
                # Packed NF4 base weights have uint8 storage, not the adapter dtype.
                for module in self.model.modules():
                    if isinstance(module, _PeftLoraLayer):
                        for key in module.lora_A:
                            module.lora_A[key].float()
                            module.lora_B[key].float()
            self.lora_layers = {name: module for name, module in self.model.named_modules()
                if isinstance(module, _PeftLoraLayer)}
            self.model._lora_layers = self.lora_layers
            write_json(self.output_dir / "adapter-settings.json", {
                "variant": self.variant, "use_dora": self.variant == "dora",
                "use_rslora": self.variant == "rslora", "r": self.config.lora_r,
                "alpha": self.config.lora_alpha, "targets": targets,
                "scaling": {name: dict(module.scaling) for name, module in self.lora_layers.items()},
                "quantization": getattr(self.model, "_crm_quantization_report", None)})

        def _create_dataloader(self, dataset, batch_size, shuffle=True, is_training=True):
            size = min(batch_size, len(dataset))
            if size < 1:
                raise ValueError("Empty dataset")
            sampler = None
            if shuffle:
                sampler = LengthGroupedSampler(dataset.lengths, size,
                    window_batches=self.config.length_group_window_batches,
                    seed=self.config.seed)
            return DataLoader(dataset, batch_size=size, sampler=sampler,
                shuffle=False, num_workers=0, pin_memory=self.device.type == "cuda",
                drop_last=False, collate_fn=StrictCollator(
                    self.processor, self.fixed_schema, self.max_encoded_tokens))

        def _backward_one(self, batch, *args, **kwargs):
            try:
                value = super()._backward_one(batch, *args, **kwargs)
            except torch.cuda.OutOfMemoryError as error:
                # Official 2.1.0 train() otherwise catches OOM and continues with
                # missing examples. A failed resource probe must remain failed.
                raise RuntimeError("CUDA OOM: stopped without silently dropping training data") from error
            self._flush_delayed_counters()
            self.processed_examples += len(batch)
            self.successful_microbatches += 1
            return value

        def _optimizer_step(self):
            # Both configured modes use BF16 or FP32, without a GradScaler.
            self._flush_delayed_counters()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm,
                error_if_nonfinite=True)
            self._last_grad_norm = norm.detach()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            return True

        def _create_optimizer(self):
            optimizer = super()._create_optimizer()
            names = {id(p): name for name, p in self.model.named_parameters()}
            groups = []
            for group in optimizer.param_groups:
                groups.append({"lr": group["lr"], "weight_decay": group["weight_decay"],
                    "parameters": [{"name": names[id(p)], "numel": p.numel()} for p in group["params"]]})
            write_json(self.output_dir / "optimizer-groups.json", groups)
            return optimizer

        def _log_metrics(self, metrics, prefix=""):
            super()._log_metrics(metrics, prefix)
            value = metrics if isinstance(metrics, dict) else metrics.to_dict()
            with (self.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(json_safe({"prefix": prefix, **value})) + "\n")

        def _save_checkpoint(self, name):
            name = name.replace("checkpoint-epoch-", "epoch-")
            path = self.output_dir / name
            if path.exists():
                raise FileExistsError(path)
            start = time.perf_counter()
            super()._save_checkpoint(name)
            elapsed = time.perf_counter() - start
            metadata = {"epoch": self.epoch + 1, "optimizer_steps": self.global_step,
                "examples_processed": self.processed_examples, "save_seconds": elapsed,
                "checkpoint_type": "adapter" if self.config.use_lora else "full",
                "exact_optimizer_resume_supported": False,
                "files": {str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                          for p in path.rglob("*") if p.is_file()}}
            write_json(path / "checkpoint.json", metadata)
            self.saved_checkpoints.append({"path": name, **metadata})
            self.checkpoint_seconds += time.perf_counter() - start

    return RentalTrainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pilot-steps", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Train-only bounded smoke/pilot subset; requires --pilot-steps")
    parser.add_argument("--precision", choices=("bf16", "fp32", "float32"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--reload-smoke", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.pilot_steps < 0 or args.max_train_samples < 0:
        raise ValueError("Pilot steps and maximum samples must be nonnegative")
    if args.max_train_samples and not args.pilot_steps:
        raise ValueError("Subsetting is allowed only for an explicitly marked pilot/smoke")
    root = args.root.resolve()
    cfg = load_json(args.config)
    mode = cfg["mode"]
    if mode not in ("full", "lora", "qlora", "dora", "rslora"):
        raise ValueError("Unsupported training variant")
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    report = {"status": "running", "mode": mode, "pilot_steps": args.pilot_steps,
        "run_kind": "pilot" if args.pilot_steps else "main",
        "run_type": "pilot" if args.pilot_steps else "main", "device": args.device,
        "argv": sys.argv, "config": cfg, "start_unix": time.time(),
        "training_data": "train only", "test_inference_calls": 0,
        "exact_optimizer_resume_supported": False}
    write_json(out / "run.json", report)
    file_log = logging.FileHandler(out / "training.log")
    file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(file_log)
    logging.getLogger().setLevel(logging.INFO)
    started = time.perf_counter()
    trainer = None
    try:
        import torch
        from gliner2.training.trainer import TrainingConfig
        torch.set_num_threads(args.cpu_threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        precision = args.precision or cfg.get("precision", "bf16")
        if precision == "float32":
            precision = "fp32"
        if args.device == "cpu" and precision != "fp32":
            raise ValueError("CPU smoke requires explicit --precision fp32")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; choose an explicitly recorded FP32 configuration")
        if precision not in ("bf16", "fp32"):
            raise ValueError("Only bf16/fp32 are supported by this strict wrapper")
        batch_size = args.batch_size or cfg["batch_size"]
        accumulation = args.gradient_accumulation_steps or cfg["gradient_accumulation_steps"]
        if batch_size < 1 or accumulation < 1:
            raise ValueError("Batch and gradient accumulation must be positive")
        seed_all(cfg["seed"])
        report["data_verification"] = verify_data(root)
        from common import BASE_SHA256
        if sha(root / "model/model.safetensors") != BASE_SHA256:
            raise ValueError("Base model weights differ from the frozen checkpoint")
        data = root / "data/corpus-v1"
        raw = read_jsonl(data / "train.jsonl")
        exports = read_jsonl(data / "gliner/train.jsonl")
        if len(raw) != len(exports):
            raise ValueError("Raw/export lengths differ")
        schema = load_json(data / "model-label-schema.json")
        indices = list(range(len(raw)))
        if args.pilot_steps:
            lengths = {r["id"]: r["encoded_tokens_including_schema"]
                       for r in read_jsonl(data / "tokenization-train.jsonl")}
            if args.device == "cpu" and args.max_train_samples:
                # Tiny local correctness check; GPU pilots still include long inputs.
                indices = sorted(indices, key=lambda i: (lengths[raw[i]["id"]], raw[i]["id"]))[:args.max_train_samples]
                report["pilot_sampling"] = "shortest train records for bounded CPU correctness smoke"
            else:
                indices = choose_pilot_indices(raw, lengths, args.max_train_samples or 64)
                report["pilot_sampling"] = "train-only long/negative/mixed/all-label coverage"
        records = [exports[i] for i in indices]
        write_json(out / "training-records.json", {"ids": [raw[i]["id"] for i in indices],
            "split": "train", "sha256": sha(data / "train.jsonl")})
        source = out / "source"
        source.mkdir()
        for filename in ("training.py", "common.py", "quantization.py"):
            shutil.copy2(Path(__file__).with_name(filename), source / filename)
        model = load_model(root / "model", device=args.device, variant=mode)
        if getattr(model, "architecture", None) != "boundary":
            raise ValueError("Expected GLiNER2.5 boundary architecture")
        if mode == "full":
            for parameter in model.parameters():
                parameter.requires_grad = True
        else:
            targets = cfg.get("lora_target_modules", ["encoder.query", "encoder.value"])
            if targets != ["encoder.query", "encoder.value"]:
                raise ValueError("This frozen comparison expects encoder query/value LoRA only")
        options = dict(output_dir=str(out), experiment_name=out.name,
            num_epochs=cfg["num_epochs"], max_steps=args.pilot_steps or -1,
            batch_size=batch_size, gradient_accumulation_steps=accumulation,
            encoder_lr=cfg["encoder_lr"], task_lr=cfg["task_lr"],
            weight_decay=cfg.get("weight_decay", 0.01), max_grad_norm=cfg.get("max_grad_norm", 1.0),
            scheduler_type="linear", warmup_ratio=cfg.get("warmup_ratio", 0.1),
            fp16=False, bf16=precision == "bf16", seed=cfg["seed"], deterministic=True,
            eval_strategy="no" if args.pilot_steps else "epoch", save_best=False,
            save_total_limit=0, early_stopping=False, report_to_wandb=False,
            num_workers=0, pin_memory=args.device == "cuda", max_len=None,
            strict_training=True, skip_step_errors=False, allow_invalid_samples=False,
            ignore_nonfinite_losses=False, logging_steps=1, use_lora=mode != "full",
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
            compile_model=False, fused_optimizer=args.device == "cuda",
            allow_tf32=False, float32_matmul_precision="highest",
            lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            lora_target_modules=cfg.get("lora_target_modules", ["encoder.query", "encoder.value"]),
            save_adapter_only=True)
        trainer = make_trainer_class()(model, TrainingConfig(**options),
            requested_device=args.device, schema=schema,
            max_encoded_tokens=cfg.get("max_encoded_tokens", 1024), variant=mode)
        trainable = [{"name": name, "numel": p.numel(), "shape": list(p.shape), "dtype": str(p.dtype)}
                     for name, p in trainer.model.named_parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters")
        if any(p.dtype != torch.float32 for p in trainer.model.parameters() if p.requires_grad):
            raise ValueError("All trainable adapter parameters must remain FP32")
        if mode != "full" and any(not any(token in p["name"] for token in ("lora_A", "lora_B", "lora_magnitude_vector")) for p in trainable):
            raise ValueError("Unexpected unfrozen base parameters in LoRA")
        write_json(out / "trainable-parameters.json", trainable)
        probe = {name: p.detach().flatten()[:128].cpu().clone()
                 for name, p in trainer.model.named_parameters() if p.requires_grad}
        report.update({"precision": precision, "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch_size * accumulation,
            "training_examples": len(records), "train_documents": len(records),
            "epochs": cfg["num_epochs"],
            "trainable_parameters": sum(p["numel"] for p in trainable),
            "total_parameters": sum(p.numel() for p in trainer.model.parameters()),
            "gpu_validated": False, "torch": torch.__version__})
        if mode == "qlora":
            quantization = getattr(trainer.model, "_crm_quantization_report")
            report["quantization"] = quantization
            report["logical_base_parameters"] = quantization["original_parameter_count"]
            report["parameter_count_note"] = "total_parameters counts packed tensor elements; use logical_base_parameters for architecture size"
        write_json(out / "run.json", report)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        training_start = time.perf_counter()
        result = trainer.train(train_data=records)
        if args.device == "cuda":
            torch.cuda.synchronize()
        report["training_seconds_including_checkpoint"] = time.perf_counter() - training_start
        changes = {name: float((p.detach().flatten()[:128].cpu() - probe[name]).abs().max())
                   for name, p in trainer.model.named_parameters() if name in probe}
        if not any(value > 0 for value in changes.values()):
            raise RuntimeError("No sampled trainable parameter changed")
        write_json(out / "sampled-parameter-changes.json", changes)
        write_json(out / "trainer-result.json", json_safe(result))
        report.update({"optimizer_steps": trainer.global_step,
            "examples_processed": trainer.processed_examples,
            "successful_microbatches": trainer.successful_microbatches,
            "checkpoint_seconds": trainer.checkpoint_seconds,
            "checkpoints": trainer.saved_checkpoints,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None})
        if args.pilot_steps and trainer.global_step != args.pilot_steps:
            raise RuntimeError("Pilot did not complete the requested optimizer steps")
        if not args.pilot_steps and trainer.processed_examples != len(records) * cfg["num_epochs"]:
            raise RuntimeError("Not all training records were processed in every epoch")
        if args.reload_smoke:
            # Run exactly one train example, not dev/test; compare saved/reloaded
            # predictions with autocast off and eval/dropout disabled. QLoRA's
            # NF4 linear kernels still use their configured BF16 compute dtype.
            probe_text = records[0]["input"]
            trainer.model.eval()
            with torch.inference_mode():
                before = trainer.model.extract_entities(probe_text, schema["descriptions"],
                    threshold=0.5, include_spans=True, include_confidence=True, overlap_policy="flat")
            trainer.model.to("cpu")
            trainer.model = None
            trainer.optimizer = None
            trainer.scheduler = None
            del model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            reloaded = load_model(root / "model" if mode != "full" else out / "final",
                device=args.device, adapter_path=out / "final" if mode != "full" else None, variant=mode)
            reloaded.eval()
            with torch.inference_mode():
                after = reloaded.extract_entities(probe_text, schema["descriptions"],
                    threshold=0.5, include_spans=True, include_confidence=True, overlap_policy="flat")
            # FP32 merge may differ slightly in confidence. Require identical
            # entities/boundaries and tightly matching scores, not JSON bitwise identity.
            def signatures(prediction):
                return sorted((label, x["start"], x["end"], x["text"], float(x["confidence"]))
                    for label, values in prediction["entities"].items() for x in values)
            b, a = signatures(before), signatures(after)
            equivalent = len(b) == len(a) and all(x[:4] == y[:4] and abs(x[4]-y[4]) <= 1e-4 for x, y in zip(b, a))
            write_json(out / "reload-smoke.json", {"train_id": raw[indices[0]]["id"],
                "before": before, "after": after, "equivalent": equivalent, "confidence_atol": 1e-4})
            if not equivalent:
                raise RuntimeError("Save/reload prediction equivalence failed")
            report["reload_smoke_passed"] = True
        if not args.pilot_steps:
            def portable(path):
                try:
                    return str(Path(path).relative_to(root))
                except ValueError:
                    return str(Path(path))
            candidates = [{"name": item["path"], "epoch": item["epoch"],
                "model_path": portable(root / "model" if mode != "full" else out / item["path"]),
                "adapter_path": portable(out / item["path"]) if mode != "full" else None,
                "run_manifest": portable(out / "run.json")}
                for item in report["checkpoints"] if item["path"].startswith("epoch-")]
            write_json(out / "candidates.json", candidates)
        report["gpu_validated"] = args.device == "cuda"
        report["status"] = "completed"
    except BaseException as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        report["end_unix"] = time.time()
        write_json(out / "run.json", json_safe(report))
        logging.getLogger().removeHandler(file_log)
        file_log.close()
    print(json.dumps({k: report.get(k) for k in ("status", "mode", "device", "optimizer_steps",
        "examples_processed", "training_seconds_including_checkpoint", "peak_allocated_bytes")}, indent=2))


if __name__ == "__main__":
    main()
