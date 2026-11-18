#!/usr/bin/env python3
"""Generate sampler_vectors.json for tests/duplex_sampler_test.cpp.

This is an independent plain-Python implementation of the reference's rules in
.haru-ref/sampling.py and the order of operations in .haru-ref/session.py:148-165. It is not a
transcription of the C++: DRY, top-k/top-p, the float softmax's arithmetic and the inverse CDF
are written here from the reference. Standard library only; run it from the repo root:

    python3 tests/data/haru/gen_sampler_vectors.py

The C++ test uses the writes:

  penalty_cases  replay `observe` and check the lane's history and penalty map.
  sample_cases   replay `observe`, then call the injected-uniform overload with `logits` and
                 `pad_eop`, checking the token, the in-place logits, and the penalties before and
                 after the history update.

Every sample case uses the injected draw, so the lane's splitmix64 state must stay at `seed`.
"""
from __future__ import annotations

import json
import math
import os

DEFAULT_DRY = {
    "multiplier": 0.8,
    "base": 1.75,
    "allowed_length": 1,
    "lookback": 1024,
    "max_match_length": 32,
    "max_penalty": 32.0,
}


def history_size(dry):
    return dry["lookback"] + dry["max_match_length"] + 1


def dry_penalties(history, dry):
    """The reference's dry_penalties (.haru-ref/sampling.py:26-49)."""
    n = len(history)
    if n < max(1, dry["allowed_length"]):
        return {}
    longest = {}
    last = history[-1]
    for j in range(n - 2, max(0, n - 1 - dry["lookback"]) - 1, -1):
        if history[j] != last:
            continue
        length = 1
        while (
            length < dry["max_match_length"]
            and j - length >= 0
            and history[j - length] == history[n - 1 - length]
        ):
            length += 1
        if length >= dry["allowed_length"]:
            candidate = history[j + 1]
            if length > longest.get(candidate, 0):
                longest[candidate] = length
    return {
        token: min(
            dry["max_penalty"],
            dry["multiplier"] * dry["base"] ** (length - dry["allowed_length"]),
        )
        for token, length in longest.items()
    }


def observe(history, token, exempt, dry):
    if token in exempt:
        return list(history)
    h = list(history) + [token]
    extra = len(h) - history_size(dry)
    if extra > 0:
        h = h[extra:]
    return h


def prepare(logits, dry_cfg, pad, eop, eos, pad_eop, eop_bias, force_left, penalties):
    """session.py:148-156 up to but not including sample_token."""
    row = list(logits)
    for token, penalty in penalties.items():
        row[token] -= penalty
    if 0 <= pad < len(row):
        row[pad] = pad_eop[0]
    if 0 <= eop < len(row):
        row[eop] = pad_eop[1]
    if eop_bias and 0 <= eop < len(row):
        row[eop] += eop_bias
    if force_left > 0:
        banned = {pad, eop} | ({eos} if eos >= 0 else set())
        for token in banned:
            if 0 <= token < len(row):
                row[token] = -math.inf
        force_left -= 1
    return row, force_left


def sample_token(logits, temperature, top_k, top_p, uniform):
    """sampling.py:79-99 plus inverse CDF over the kept descending probabilities."""
    n = len(logits)
    if temperature <= 0:
        best = 0
        for i in range(1, n):
            if logits[i] > logits[best]:
                best = i
        return best
    values = [v / temperature for v in logits]
    order = list(range(n))
    order.sort(key=lambda i: (-values[i], i))
    if 0 < top_k < n:
        order = order[:top_k]
    top = values[order[0]]
    weights = [math.exp(values[i] - top) for i in order]
    total = sum(weights)
    probs = [w / total for w in weights]
    if 0 < top_p < 1:
        previous = 0.0
        for k, p in enumerate(probs):
            if not previous < top_p:
                probs[k] = 0.0
            previous += p
    kept = [(order[k], probs[k]) for k in range(len(order)) if probs[k] > 0.0]
    total = sum(p for _, p in kept)
    target = uniform * total
    acc = 0.0
    for token, p in kept:
        acc += p
        if target < acc:
            return token
    return kept[-1][0]


def enc(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def penalty_case(name, observed, dry=None, pad=900, eop=901, eos=-1):
    dry = dict(DEFAULT_DRY if dry is None else {**DEFAULT_DRY, **dry})
    exempt = {pad, eop} | ({eos} if eos >= 0 else set())
    history = []
    for token in observed:
        history = observe(history, token, exempt, dry)
    penalties = dry_penalties(history, dry)
    penalties = {t: p for t, p in penalties.items() if t not in exempt}
    return {
        "name": name,
        "dry": dry,
        "pad_id": pad,
        "eop_id": eop,
        "eos_id": eos,
        "observe": observed,
        "expected_history": history,
        "expected_penalties": {str(k): v for k, v in sorted(penalties.items())},
    }


def sample_case(
    name,
    logits,
    *,
    pad,
    eop,
    eos,
    pad_eop,
    uniform,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    eop_bias=0.0,
    dry=None,
    observed=(),
    force_text_frames=0,
    seed=1234,
):
    dry = dict(DEFAULT_DRY if dry is None else {**DEFAULT_DRY, **dry})
    exempt = {pad, eop} | ({eos} if eos >= 0 else set())
    history = []
    for token in observed:
        history = observe(history, token, exempt, dry)
    penalties_before = {
        t: p for t, p in dry_penalties(history, dry).items() if t not in exempt
    }
    row, force_left = prepare(
        logits, dry, pad, eop, eos, pad_eop, eop_bias, force_text_frames, penalties_before
    )
    token = sample_token(row, temperature, top_k, top_p, uniform)
    history_after = observe(history, token, exempt, dry)
    penalties_after = {
        t: p for t, p in dry_penalties(history_after, dry).items() if t not in exempt
    }
    return {
        "name": name,
        "logits": list(logits),
        "pad_eop": list(pad_eop),
        "pad_id": pad,
        "eop_id": eop,
        "eos_id": eos,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "eop_bias": eop_bias,
        "dry": dry,
        "observe": list(observed),
        "force_text_frames": force_text_frames,
        "seed": seed,
        "uniform": uniform,
        "expected_logits": [enc(v) for v in row],
        "expected_token": token,
        "expected_penalties_before": {str(k): v for k, v in sorted(penalties_before.items())},
        "expected_penalties_after": {str(k): v for k, v in sorted(penalties_after.items())},
        "expected_history_after": history_after,
        "expected_force_text_after": force_left,
        "expected_rng_state": seed,
    }


def main():
    penalty_cases = [
        penalty_case("empty", []),
        penalty_case("below_allowed_length", [1, 2], dry={"allowed_length": 3}),
        penalty_case("match_length_1", [7, 7]),
        penalty_case("match_length_2", [10, 20, 10, 20]),
        penalty_case("match_length_3", [10, 20, 10, 20, 10]),
        penalty_case("several_candidates", [7, 8, 9, 7, 9, 7]),
        penalty_case("max_penalty_clamp", [42] * 40),
        penalty_case("exempt_tokens", [1, 2, 1, 3, 1, 4, 1], pad=2, eop=3, eos=4),
        penalty_case(
            "history_size_trim",
            [100, 1, 2, 1, 3, 1, 2, 1, 3],
            dry={"lookback": 4, "max_match_length": 2},
        ),
    ]

    sample_cases = [
        sample_case(
            "greedy_ties_and_replacement",
            [1.0, 3.0, 3.0, 10.0, 10.0],
            pad=3,
            eop=4,
            eos=-1,
            pad_eop=[2.0, 0.5],
            uniform=0.0,
            temperature=0.0,
        ),
        sample_case(
            "top_k_1",
            [0.5, 2.0, 1.0, -1.0, 5.0, 5.0],
            pad=4,
            eop=5,
            eos=-1,
            pad_eop=[-2.0, -3.0],
            uniform=0.99,
            top_k=1,
        ),
        sample_case(
            "top_k_above_vocab",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 50.0, 50.0],
            pad=8,
            eop=9,
            eos=-1,
            pad_eop=[-5.0, -6.0],
            uniform=0.3,
            top_k=25,
        ),
        sample_case(
            "top_p_below_one",
            [2.0, 1.0, 0.0, -1.0, -2.0, 9.0, 9.0],
            pad=5,
            eop=6,
            eos=-1,
            pad_eop=[-9.0, -9.0],
            uniform=0.85,
            top_k=0,
            top_p=0.8,
        ),
        sample_case(
            "top_p_strict_and_ties",
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            pad=4,
            eop=5,
            eos=-1,
            pad_eop=[1.0, 1.0],
            uniform=0.6,
            top_k=0,
            top_p=0.5,
        ),
        sample_case(
            "top_k_tie_cut",
            [5.0, 5.0, 5.0, 5.0],
            pad=2,
            eop=3,
            eos=-1,
            pad_eop=[5.0, 5.0],
            uniform=0.95,
            top_k=3,
        ),
        sample_case(
            "force_text_bans_eos",
            [100.0, 100.0, 100.0, 2.0, 1.0, 0.0],
            pad=0,
            eop=1,
            eos=2,
            pad_eop=[100.0, 100.0],
            uniform=0.0,
            force_text_frames=1,
        ),
        sample_case(
            "force_text_without_eos",
            [100.0, 100.0, 100.0, 2.0, 1.0, 0.0],
            pad=0,
            eop=1,
            eos=-1,
            pad_eop=[100.0, 100.0],
            uniform=0.0,
            force_text_frames=1,
        ),
        sample_case(
            "pad_eop_bias_after_replacement",
            [10.0, 0.0, 1.0, 2.0, 3.0],
            pad=0,
            eop=1,
            eos=-1,
            pad_eop=[0.0, 2.0],
            uniform=0.0,
            temperature=0.0,
            eop_bias=5.0,
        ),
        sample_case(
            "dry_changes_token",
            [0.0, 2.5, 2.0, 0.0, 0.0],
            pad=3,
            eop=4,
            eos=-1,
            pad_eop=[-5.0, -5.0],
            uniform=0.0,
            temperature=0.0,
            observed=[2, 1, 2],
        ),
        sample_case(
            "dry_match_length_2",
            [0.0, 2.0, 1.8, 0.0, 0.0],
            pad=3,
            eop=4,
            eos=-1,
            pad_eop=[-5.0, -5.0],
            uniform=0.0,
            temperature=0.0,
            observed=[1, 2, 1, 2],
        ),
    ]

    data = {
        "source": "tests/data/haru/gen_sampler_vectors.py",
        "penalty_cases": penalty_cases,
        "sample_cases": sample_cases,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampler_vectors.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
