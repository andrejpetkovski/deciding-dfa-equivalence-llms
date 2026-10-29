"""Post-review structural audit and descriptive family/rendering performance."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

from src.io_utils import read_jsonl


def rooted_mapping(a: dict, b: dict) -> dict[int, int]:
    """Recover the unique reachable transition-preserving mapping, ignoring acceptance."""
    mapping = {a["start"]: b["start"]}
    inverse = {b["start"]: a["start"]}
    queue = deque(mapping)
    while queue:
        state = queue.popleft()
        for target, other in zip(a["delta"][state], b["delta"][mapping[state]], strict=True):
            if target in mapping:
                if mapping[target] != other:
                    raise ValueError("Transitions disagree")
            else:
                if other in inverse:
                    raise ValueError("Mapping is not bijective")
                mapping[target] = other
                inverse[other] = target
                queue.append(target)
    if len(mapping) != a["n_states"] or len(inverse) != b["n_states"]:
        raise ValueError("Mapping does not cover both machines")
    return mapping


def performance(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["sample_index"] == 0:
            groups[tuple(row[f] for f in fields)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        neq = group[0]["family"] == "NEQ_DEPTH"
        count = sum(bool(r["witness_success"] if neq else r["decision_correct"]) for r in group)
        result.append(
            {
                **dict(zip(fields, key, strict=True)),
                "successes": count,
                "calls": len(group),
                "rate": count / len(group),
                "metric": "valid_witness" if neq else "decision_accuracy",
            }
        )
    return result


def main() -> None:
    paths = [
        Path(p)
        for p in (
            "data/pairs/main_v2/pairs.jsonl",
            "data/renderings/main_v2/manifest.jsonl",
            "data/parsed/main_v2/graded.jsonl",
        )
    ]
    pairs, manifest, rows = (list(read_jsonl(p)) for p in paths)
    manifests = {(r["pair_id"], r["rendering"]): r for r in manifest}
    audits = []
    for pair in pairs:
        if pair["family"] != "NEQ_DEPTH":
            continue
        a, b = pair["dfa_A"], pair["dfa_B"]
        mapping = rooted_mapping(a, b)
        mismatches = sum((s in a["accepting"]) != (t in b["accepting"]) for s, t in mapping.items())
        assert mismatches == 2
        entry = {
            "pair_id": pair["pair_id"],
            "acceptance_mismatches": mismatches,
            "identical_stored_tables": a["delta"] == b["delta"],
        }
        for rendering in ("R0", "R1"):
            m = manifests[(pair["pair_id"], rendering)]
            entry[f"{rendering}_all_corresponding_labels_equal"] = all(
                m["labels_A"][s] == m["labels_B"][t] for s, t in mapping.items()
            )
            texts = [
                Path(f"data/renderings/main_v2/{pair['pair_id']}_{name}_{rendering}.txt").read_text(
                    encoding="utf-8"
                )
                for name in ("A", "B")
            ]
            transitions = [
                {line.strip() for line in text.splitlines() if "->" in line} for text in texts
            ]
            assert all(len(t) == 28 for t in transitions)
            entry[f"{rendering}_identical_displayed_transitions"] = transitions[0] == transitions[1]
        audits.append(entry)
    report = {
        "status": "Post-review descriptive analysis; primary grades unchanged",
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "structural_audit": audits,
        "by_family_rendering": performance(rows, ("family", "rendering")),
        "by_family_rendering_model_prompt": performance(
            rows, ("family", "rendering", "model_id", "prompt_condition")
        ),
    }
    out = Path("analysis/tables/review_20261028")
    out.mkdir(parents=True, exist_ok=True)
    (out / "construction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Construction and family/rendering audit",
        "",
        f"Audited {len(audits)} non-equivalent pairs. Every pair has a rooted transition "
        "bijection with exactly two acceptance mismatches.",
        "",
    ]
    for key in audits[0]:
        if key not in ("pair_id", "acceptance_mismatches"):
            lines.append(f"- {key}: {sum(r[key] for r in audits)}/{len(audits)}")
    lines += [
        "",
        "R0 uses stored order after independent A/B permutations in src/generate.py. "
        "R1 additionally uses separate random streams keyed by pair seed, rendering, "
        "and A/B name in src/render_prompts.py. Shared label vocabulary does not imply "
        "shared state correspondence. A transition alignment still exposes the two "
        "acceptance changes; this construction does not cover general NEQ structure.",
        "",
        "All rates below retain parse failures as errors and use primary sample 0 only. "
        "They are descriptive; they do not establish a causal rendering effect.",
        "",
        "| Family | Rendering | Metric | Successes / calls | Rate |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["by_family_rendering"]:
        lines.append(
            f"| {row['family']} | {row['rendering']} | {row['metric']} | "
            f"{row['successes']}/{row['calls']} | {100 * row['rate']:.2f}% |"
        )
    lines += [
        "",
        "Model/prompt intersections and per-pair structural checks are in "
        "construction.json. Reproduce with `python -m analysis.review_construction`.",
        "",
        "Different-structure non-equivalent pairs would require a separate experiment. "
        "A useful extension is an acceptance change in one member of an EQ_EXPAND pair, "
        "followed by oracle verification, acceptance-count controls, and depth matching. "
        "No such experiment was run, and its outcomes are not claimed here.",
        "",
    ]
    (out / "construction.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
