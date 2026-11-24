# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Small, reproducible DCVC-RT fine-tuning experiments on Vimeo septuplets.

Based on root train_video.py's frozen I codec, clip RD, gradient clipping and
checkpoint workflow. This script adds the missing differentiable RT path. Its
defaults describe a limited fine-tuning experiment, not full paper training.
UVG/HEVC must remain external held-out bitstream benchmarks.

Example (use the free, authorized GPU's UUID for unambiguous mapping):
  CUDA_VISIBLE_DEVICES=GPU-... python train_video_rt.py --device cuda:0 \
    --dataset-root @VIMEO_ROOT@ \
    --model-path-i @RT_WEIGHTS@/cvpr2025_image.pth.tar \
    --pretrain-path @RT_WEIGHTS@/cvpr2025_video.pth.tar \
    --save-dir runs/rt_finetune_01 --epochs 3 --steps-per-epoch 50
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.training_model import IntraForTraining, VideoForTraining
from src.utils.transforms import rgb2ycbcr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path-i", type=Path, required=True)
    parser.add_argument("--pretrain-path", "--model-path-p", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0", help="Visible CUDA index, restricted to physical GPU 4/5; cpu for checks")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--train-clips", type=int, default=2048, help="0 selects the complete official training list")
    parser.add_argument("--val-clips", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=0.2)
    parser.add_argument("--qp", type=int, nargs="+", default=[0, 21, 42, 63])
    parser.add_argument("--lambda-min", type=float, default=85.0)
    parser.add_argument("--lambda-max", type=float, default=7680.0)
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    parser.add_argument("--cascade", action="store_true", help="Keep gradients through reference frames across the clip")
    parser.add_argument("--rgb444", action="store_true", help="Retain 444 chroma; default emulates YUV420 chroma sampling")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.frames <= 7:
        parser.error("--frames must be between 2 and 7")
    if args.crop_size not in (64, 128, 192, 256):
        parser.error("--crop-size must be a multiple of 64, between 64 and 256")
    for name in ("epochs", "steps_per_epoch", "batch_size", "val_clips", "log_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_clips < 0 or args.num_workers < 0:
        parser.error("train-clips and num-workers cannot be negative")
    if not args.lr > 0 or not args.grad_clip > 0:
        parser.error("learning rate and gradient clip must be positive")
    if not 0 < args.lambda_min <= args.lambda_max:
        parser.error("Require 0 < lambda-min <= lambda-max")
    if not math.isfinite(args.force_zero_thres) or not 0 <= args.force_zero_thres < 16:
        parser.error("--force-zero-thres must be finite and in [0,16)")
    if any(qp < 0 or qp > 63 for qp in args.qp) or len(set(args.qp)) != len(args.qp):
        parser.error("--qp requires unique base values in [0,63]")
    for name in ("dataset_root", "model_path_i", "pretrain_path", "save_dir", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def authorized_device(requested):
    """Fail closed unless all CUDA-visible devices map to physical GPU 4 or 5."""
    device = torch.device(requested)
    if device.type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return device, {"device": "cpu"}
    if device.type != "cuda":
        raise ValueError("Only cpu or cuda devices are supported")
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visibility.split(",") if token.strip()]
    if not tokens:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the UUID or physical index of authorized GPU 4/5")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the physical GPU restriction was checked")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    inventory = {}
    for line in query.stdout.strip().splitlines():
        index, uuid = [value.strip() for value in line.split(",", 1)]
        inventory[int(index)] = uuid
    resolved = []
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if index not in inventory:
                raise RuntimeError(f"GPU index {index} is absent")
        else:
            matches = [index for index, uuid in inventory.items() if uuid.startswith(token)]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot resolve visible GPU token {token!r}")
            index = matches[0]
        if index not in (4, 5):
            raise RuntimeError(f"Physical GPU {index} is forbidden; only GPU 4 and 5 are authorized")
        resolved.append(index)
    if len(resolved) != len(set(resolved)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES contains duplicate devices")
    # Normalize indices to UUIDs before the first CUDA initialization.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(inventory[index] for index in resolved)
    visible_index = device.index if device.index is not None else 0
    if visible_index >= len(resolved):
        raise RuntimeError("Requested CUDA index exceeds CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected environment")
    torch.cuda.set_device(visible_index)
    return torch.device("cuda", visible_index), {
        "device": f"cuda:{visible_index}", "physical_gpu": resolved[visible_index],
        "gpu_uuid": inventory[resolved[visible_index]],
        "gpu_name": torch.cuda.get_device_name(visible_index)}


def seed_everything(seed, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also calls the CUDA seeding API. Keep CPU checks entirely
    # on the CPU, including when a CUDA-enabled Torch build is installed.
    torch.random.default_generator.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def read_official_list(path):
    if not path.is_file():
        raise FileNotFoundError(f"Official Vimeo split list is required: {path}")
    clips = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip().replace("\\", "/")
        if not value:
            continue
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValueError(f"Invalid Vimeo clip identifier in {path}: {value!r}")
        clips.append(relative.as_posix())
    if not clips or len(set(clips)) != len(clips):
        raise ValueError(f"Empty split or duplicate clip identifiers: {path}")
    return clips


def list_digest(values):
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_splits(root, train_count, val_count, seed):
    train = read_official_list(root / "sep_trainlist.txt")
    validation = read_official_list(root / "sep_testlist.txt")
    overlap = set(train).intersection(validation)
    if overlap:
        raise ValueError(f"Official train/test lists overlap ({len(overlap)} clips)")
    full_train_hash, full_val_hash = list_digest(train), list_digest(validation)
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    if train_count:
        train = train[:train_count]
    validation = validation[:val_count]
    return train, validation, {"official_train_sha256": full_train_hash,
                               "official_test_sha256": full_val_hash,
                               "selected_train": train, "selected_validation": validation,
                               "selected_train_sha256": list_digest(train),
                               "selected_validation_sha256": list_digest(validation),
                               "split_policy": "Vimeo official sep_trainlist/sep_testlist; UVG/HEVC excluded"}


class VimeoClips(Dataset):
    def __init__(self, root, clips, frames, crop_size, training, seed, chroma420=True):
        self.root = root / "sequences"
        self.clips = clips
        self.frames = frames
        self.crop_size = crop_size
        self.training = training
        self.seed = seed
        self.chroma420 = chroma420
        self.epoch = 0
        for clip in clips:
            folder = self.root.joinpath(*PurePosixPath(clip).parts)
            if any(not (folder / f"im{index}.png").is_file() for index in range(1, 8)):
                raise FileNotFoundError(f"Vimeo septuplet is incomplete: {folder}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        rng = random.Random(self.seed + self.epoch * 1000003 + index)
        start = rng.randrange(8 - self.frames) if self.training else 0
        indices = list(range(start + 1, start + self.frames + 1))
        if self.training and rng.random() < 0.5:
            indices.reverse()
        flip = self.training and rng.random() < 0.5
        folder = self.root.joinpath(*PurePosixPath(self.clips[index]).parts)
        images = []
        crop = None
        expected_size = None
        for frame_index in indices:
            with Image.open(folder / f"im{frame_index}.png") as source:
                image = source.convert("RGB")
                width, height = image.size
                if expected_size is None:
                    expected_size = image.size
                if image.size != expected_size:
                    raise ValueError(f"Inconsistent frame sizes in {folder}")
                if width < self.crop_size or height < self.crop_size:
                    raise ValueError(f"Clip smaller than crop size: {folder}")
                if crop is None:
                    if self.training:
                        left = rng.randrange(width - self.crop_size + 1)
                        top = rng.randrange(height - self.crop_size + 1)
                    else:
                        left = (width - self.crop_size) // 2
                        top = (height - self.crop_size) // 2
                    crop = (left, top, left + self.crop_size, top + self.crop_size)
                image = image.crop(crop)
                if flip:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                array = np.asarray(image, dtype=np.float32).copy() / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        frames = rgb2ycbcr(torch.stack(images))
        if self.chroma420:
            chroma = F.interpolate(F.avg_pool2d(frames[:, 1:], 2, 2), scale_factor=2, mode="nearest")
            frames = torch.cat((frames[:, :1], chroma), 1)
        return frames.contiguous()


def load_weights(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
        payload = payload["state_dict"]
    elif "net" in payload:
        payload = payload["net"]
    return {key.removeprefix("module."): value for key, value in payload.items()}


def atomic_save(payload, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def json_arguments(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, event, **values):
        record = {"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")
        print(line, flush=True)


def detached_stats(result):
    return {key: float(value.detach().cpu()) for key, value in result.items()}


def validation(model, intra_model, dataset, args, device, lambda_table):
    model.eval()
    totals = {}
    with torch.no_grad():
        for qp in args.qp:
            sums = {}
            count = 0
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed(args.seed))
            for frames in loader:
                frames = frames.to(device, non_blocking=True)
                batch_qp = torch.full((len(frames),), qp, device=device, dtype=torch.long)
                result = model(frames, batch_qp, intra_model, lambda_table[batch_qp],
                               detach_refs=True, noisy_rate=False)
                for key, value in detached_stats(result).items():
                    sums[key] = sums.get(key, 0.0) + value * len(frames)
                count += len(frames)
            totals[str(qp)] = {key: value / count for key, value in sums.items()}
            totals[str(qp)]["clips"] = count
    return totals


def state_for_resume(model, optimizer, args, manifest, epoch, updates, use_cuda=False):
    return {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "optimizer_updates": updates, "config": json_arguments(args),
            "selected_train_sha256": manifest["selected_train_sha256"],
            "selected_validation_sha256": manifest["selected_validation_sha256"],
            "frozen_intra_sha256": manifest["frozen_intra_sha256"],
            "pretrain_sha256": manifest["pretrain_sha256"],
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if use_cuda else []}


def main():
    args = parse_args()
    # Required for deterministic CUDA BLAS; must precede CUDA initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device, hardware = authorized_device(args.device)
    seed_everything(args.seed, use_cuda=device.type == "cuda")
    if any((args.save_dir / name).exists() for name in ("last.pth.tar", "config.json")) and args.resume is None:
        raise FileExistsError("This run already exists; use a new save-dir or explicit --resume")
    for source in (args.model_path_i, args.pretrain_path):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.parent == args.save_dir:
            raise ValueError("Use a distinct save directory to protect the pretrained checkpoints")
    train_ids, val_ids, manifest = select_splits(args.dataset_root, args.train_clips,
                                                args.val_clips, args.seed)
    manifest["frozen_intra_sha256"] = file_digest(args.model_path_i)
    manifest["pretrain_sha256"] = file_digest(args.pretrain_path)
    train_dataset = VimeoClips(args.dataset_root, train_ids, args.frames, args.crop_size,
                               True, args.seed, chroma420=not args.rgb444)
    validation_dataset = VimeoClips(args.dataset_root, val_ids, args.frames, args.crop_size,
                                    False, args.seed, chroma420=not args.rgb444)
    intra_model = IntraForTraining(force_zero_thres=args.force_zero_thres)
    intra_model.load_state_dict(load_weights(args.model_path_i), strict=True)
    intra_model.requires_grad_(False).eval().to(device)
    model = VideoForTraining(force_zero_thres=args.force_zero_thres)
    model.load_state_dict(load_weights(args.pretrain_path), strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lambda_table = torch.exp(torch.linspace(math.log(args.lambda_min), math.log(args.lambda_max),
                                            64, device=device))
    start_epoch = 0
    updates = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("selected_train_sha256", "selected_validation_sha256",
                    "frozen_intra_sha256", "pretrain_sha256"):
            if resume[key] != manifest[key]:
                raise ValueError(f"Resume split mismatch: {key}")
        for key in ("frames", "crop_size", "seed", "qp", "lambda_min", "lambda_max", "cascade",
                    "rgb444", "force_zero_thres", "batch_size", "steps_per_epoch", "lr"):
            if resume["config"][key] != getattr(args, key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = resume["epoch"] + 1
        updates = resume["optimizer_updates"]
        torch.set_rng_state(resume["torch_rng"])
        random.setstate(resume["python_rng"])
        if resume["cuda_rng"] and device.type == "cuda":
            if len(resume["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Resume CUDA visibility changed; use the original visible device count")
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    if start_epoch >= args.epochs:
        raise ValueError("--epochs is the total target epoch count and is already completed")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    config = {"arguments": json_arguments(args), "hardware": hardware,
              "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
              "training_recipe": "Adapted RT fine-tuning; not the original unpublished training recipe",
              "rate_metric": "P-frame estimated Gaussian/learned-CDF bpp, excludes I and headers",
              "distortion_metric": "normalized YUV420 weighted 6:1:1 MSE",
              "psnr_metric": "(6*PSNR_Y+PSNR_U+PSNR_V)/8 on cropped Vimeo clips"}
    (args.save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.save_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log = Logger(args.save_dir / "metrics.jsonl")
    log("run_start", **config, optimizer_updates=updates)
    baseline = validation(model, intra_model, validation_dataset, args, device, lambda_table)
    log("resume_validation" if args.resume else "baseline_validation", epoch=start_epoch - 1,
        optimizer_updates=updates, rates=baseline)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_dataset.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            generator=generator)
        iterator = iter(loader)
        qp_rng = random.Random(args.seed + epoch * 100003)
        reference = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        epoch_updates = 0
        epoch_start = time.monotonic()
        totals = {}
        for step in range(args.steps_per_epoch):
            try:
                frames = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                frames = next(iterator)
            frames = frames.to(device, non_blocking=True)
            qp = torch.tensor([qp_rng.choice(args.qp) for _ in range(len(frames))],
                              dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            result = model(frames, qp, intra_model, lambda_table[qp],
                           detach_refs=not args.cascade, noisy_rate=True)
            if not bool(torch.isfinite(result["loss"])):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch}, step {step}")
            result["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                           error_if_nonfinite=True)
            if not bool(gradient_norm > 0):
                raise FloatingPointError("All gradients are zero; refusing to record a fake optimizer update")
            optimizer.step()
            updates += 1
            epoch_updates += 1
            stats = detached_stats(result)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            if step % args.log_every == 0 or step + 1 == args.steps_per_epoch:
                log("train_step", epoch=epoch, step=step + 1, optimizer_updates=updates,
                    base_qp=qp.cpu().tolist(), gradient_norm=float(gradient_norm.cpu()),
                    lr=optimizer.param_groups[0]["lr"], **stats)
        delta_squared = sum(float((parameter.detach().cpu() - reference[name]).double().square().sum())
                            for name, parameter in model.named_parameters())
        if not delta_squared > 0:
            raise FloatingPointError("Parameters did not change despite optimizer steps")
        del reference
        metrics = validation(model, intra_model, validation_dataset, args, device, lambda_table)
        log("epoch_end", epoch=epoch, optimizer_updates=updates, epoch_optimizer_updates=epoch_updates,
            parameter_l2_delta=math.sqrt(delta_squared), elapsed_seconds=time.monotonic() - epoch_start,
            train_mean={key: value / args.steps_per_epoch for key, value in totals.items()}, rates=metrics)
        state = state_for_resume(model, optimizer, args, manifest, epoch, updates,
                                 use_cuda=device.type == "cuda")
        atomic_save(state, args.save_dir / f"checkpoint_ep{epoch + 1:03d}.pth.tar")
        atomic_save(state, args.save_dir / "last.pth.tar")
        # Original test_video.py accepts this checkpoint without adapter imports.
        atomic_save({"state_dict": model.state_dict()}, args.save_dir / "ckpt.pth.tar")
    log("run_complete", completed_epochs=args.epochs, optimizer_updates=updates,
        checkpoint=str(args.save_dir / "ckpt.pth.tar"))


if __name__ == "__main__":
    main()
