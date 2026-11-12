#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, is_dataclass
from pprint import pprint

import torch


def to_plain(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in vars(obj).items()}
    return obj


def print_state_dict_shapes(name, sd, max_lines=80):
    print(f"\n{name} parameter shapes")
    print("-" * 80)
    count = 0
    for k, v in sd.items():
        if hasattr(v, "shape"):
            print(f"{k:50s} {tuple(v.shape)}")
            count += 1
            if count >= max_lines:
                print(f"... truncated after {max_lines} tensors")
                break


def first_last_linear(sd):
    linear_weights = []
    for k, v in sd.items():
        if hasattr(v, "shape") and "weight" in k and len(v.shape) == 2:
            linear_weights.append((k, tuple(v.shape)))
    if not linear_weights:
        return None, None
    return linear_weights[0], linear_weights[-1]


def find_reward_fields(cfg_plain, prefix=""):
    rows = []

    if isinstance(cfg_plain, dict):
        for k, v in cfg_plain.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            lk = str(k).lower()

            if isinstance(v, dict):
                rows.extend(find_reward_fields(v, key))
            else:
                if any(s in lk for s in [
                    "reward", "penalty", "bonus", "collision", "timeout",
                    "fall", "progress", "smooth", "correction", "align",
                    "reverse", "yaw", "time"
                ]):
                    rows.append((key, v))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--show-all-config", action="store_true")
    parser.add_argument("--show-shapes", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    print("=" * 80)
    print(args.checkpoint)
    print("=" * 80)

    print("\nCheckpoint keys:")
    print(list(ckpt.keys()))

    print("\nTraining counters:")
    for key in ["global_step", "gradient_step"]:
        print(f"{key}: {ckpt.get(key, None)}")

    if "log_alpha" in ckpt:
        log_alpha = ckpt["log_alpha"]
        if torch.is_tensor(log_alpha):
            log_alpha_val = float(log_alpha.item())
        else:
            log_alpha_val = float(log_alpha)
        print(f"log_alpha: {log_alpha_val}")
        print(f"alpha = exp(log_alpha): {math.exp(log_alpha_val)}")

    actor_sd = ckpt.get("actor", None)
    critic_sd = ckpt.get("critic", None)

    if actor_sd is not None:
        first, last = first_last_linear(actor_sd)
        print("\nActor network:")
        print("first linear-like weight:", first)
        print("last linear-like weight: ", last)

        if first is not None:
            print("inferred actor input dim:", first[1][1])
        if last is not None:
            print("raw actor output dim:", last[1][0])
            print("likely action dim:", last[1][0] // 2)

    if critic_sd is not None:
        first, last = first_last_linear(critic_sd)
        print("\nCritic network:")
        print("first linear-like weight:", first)
        print("last linear-like weight: ", last)
        if first is not None:
            print("inferred critic input dim:", first[1][1])
        if last is not None:
            print("critic output dim:", last[1][0])

    cfg = ckpt.get("config", None)
    if cfg is not None:
        print("\nConfig type:", type(cfg))
        cfg_plain = to_plain(cfg)

        print("\nReward/config fields found:")
        rows = find_reward_fields(cfg_plain)
        if rows:
            for k, v in rows:
                print(f"{k:60s} {v}")
        else:
            print("No obvious reward-related config fields found.")

        if args.show_all_config:
            print("\nFull config:")
            pprint(cfg_plain)

    else:
        print("\nNo config stored in checkpoint.")

    if args.show_shapes:
        if actor_sd is not None:
            print_state_dict_shapes("Actor", actor_sd)
        if critic_sd is not None:
            print_state_dict_shapes("Critic", critic_sd)


if __name__ == "__main__":
    main()
