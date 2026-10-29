"""Model-free baselines for a generated pair set.

Usage::

    python -m scripts.baselines --config configs/pilot_revised.yaml

``docs/experiment_revision.md`` requires these numbers to come from a retained script before
any of them is quoted in the paper. They need no model calls: each rule looks only at the two
automata and predicts a Boolean.

**Every rule here is decision-only.** None produces a witness, so none scores above zero on
H1's primary outcome, which is end-to-end valid-witness success (D3). They bound what the
*secondary* balanced-accuracy metric can be achieved by shortcuts, and that is the claim they
support -- not a claim about the primary result.

Reported rules:

- ``constant_equivalent`` / ``constant_non_equivalent`` -- the trivial floors. A
  constant-non-equivalent rule is 100% correct on H1's restricted non-equivalent subset while
  scoring 50% balanced, which is why balanced accuracy is the metric that can be gamed and
  witness success is the one that cannot.
- ``count_only`` -- ``abs(|F_A| - |F_B|) == 1`` predicts non-equivalent. This is the cue the
  v1 single-toggle construction created and v2 exists to remove.
- ``transition_graph_iso_and_count`` -- adds "the transition graphs are isomorphic". On v1
  data this classifies every pair correctly.
- ``full_automaton_iso`` -- predicts equivalent exactly when the two machines are isomorphic
  *including* acceptance. This one survives v2 by construction: EQ_ISO pairs are isomorphic
  and the others are not. Reported openly rather than suppressed; deciding 14-state graph
  isomorphism is a far harder operation than counting two lists, and it still yields no
  witness.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from src.config import ConfigError, load_config
from src.dfa import DFA, canonical_form, from_record
from src.io_utils import canonical_json, read_jsonl, write_text
from src.oracle import equivalence_and_witness

__all__ = ["BASELINES", "evaluate", "main"]

#: Each rule maps a pair to "predict non-equivalent".
BASELINES: dict[str, Callable[[DFA, DFA], bool]] = {
    "constant_equivalent": lambda a, b: False,
    "constant_non_equivalent": lambda a, b: True,
    "count_only": lambda a, b: abs(len(a.accepting) - len(b.accepting)) == 1,
    "transition_graph_iso_and_count": lambda a, b: (
        _strip_acceptance(a) == _strip_acceptance(b)
        and abs(len(a.accepting) - len(b.accepting)) == 1
    ),
    "full_automaton_iso": lambda a, b: canonical_form(a) != canonical_form(b),
}


def _strip_acceptance(dfa: DFA) -> DFA:
    """The canonical transition structure, ignoring which states accept."""
    return canonical_form(
        DFA(
            n_states=dfa.n_states,
            alphabet=dfa.alphabet,
            start=dfa.start,
            accepting=frozenset(),
            delta=dfa.delta,
        )
    )


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score every baseline, plus an exact-oracle sanity check, over one pair set."""
    pairs = [
        (record, from_record(record["dfa_A"]), from_record(record["dfa_B"]))
        for record in records
    ]

    # Sanity check first: if the stored labels do not match a fresh oracle run, every
    # baseline number below is meaningless.
    disagreements = [
        record["pair_id"]
        for record, dfa_a, dfa_b in pairs
        if equivalence_and_witness(dfa_a, dfa_b).equivalent != record["oracle_equivalent"]
    ]

    results: dict[str, Any] = {}
    for name, rule in BASELINES.items():
        confusion = Counter()
        by_depth: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for record, dfa_a, dfa_b in pairs:
            predicted_neq = rule(dfa_a, dfa_b)
            truth_neq = not record["oracle_equivalent"]
            confusion[(truth_neq, predicted_neq)] += 1
            correct = predicted_neq == truth_neq
            family = by_family[record["family"]]
            family[0] += correct
            family[1] += 1
            if truth_neq:
                depth = by_depth[record["oracle_depth"]]
                depth[0] += correct
                depth[1] += 1

        n_neq = confusion[(True, True)] + confusion[(True, False)]
        n_eq = confusion[(False, False)] + confusion[(False, True)]
        neq_accuracy = confusion[(True, True)] / n_neq if n_neq else None
        eq_accuracy = confusion[(False, False)] / n_eq if n_eq else None
        balanced = (
            (neq_accuracy + eq_accuracy) / 2
            if neq_accuracy is not None and eq_accuracy is not None
            else None
        )
        results[name] = {
            "non_equivalent_accuracy": neq_accuracy,
            "equivalent_accuracy": eq_accuracy,
            "balanced_accuracy": balanced,
            "produces_witness": False,
            "by_family": {
                key: {"correct": v[0], "total": v[1]} for key, v in sorted(by_family.items())
            },
            "by_depth": {
                str(key): {"correct": v[0], "total": v[1]}
                for key, v in sorted(by_depth.items())
            },
        }

    return {
        "n_pairs": len(pairs),
        "oracle_sanity_check": {
            "passed": not disagreements,
            "disagreeing_pairs": disagreements[:20],
        },
        "note": (
            "All rules are decision-only and produce no witness, so all score zero on H1's "
            "primary outcome (end-to-end valid-witness success)."
        ),
        "baselines": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.baselines",
        description="Score model-free decision baselines over a generated pair set.",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        pairs_path = config.path_for("pairs_dir") / "pairs.jsonl"
        if not pairs_path.is_file():
            print(f"{pairs_path} not found. Run python -m src.generate first.",
                  file=sys.stderr)
            return 1
        records = list(read_jsonl(pairs_path))
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    report = evaluate(records)
    report["config_id"] = config.config_id
    report["construction_version"] = (
        config.get("pairs.construction_version") or "single_toggle_v1"
    )

    print(f"pair set        : {pairs_path} ({report['n_pairs']} pairs)")
    print(f"construction    : {report['construction_version']}")
    check = report["oracle_sanity_check"]
    print(f"oracle sanity   : {'PASS' if check['passed'] else 'FAIL ' + str(check)}")
    print()
    width = max(len(name) for name in BASELINES)
    print(f"  {'baseline':<{width}}  {'non-eq':>7} {'eq':>7} {'balanced':>9}")
    for name, scores in report["baselines"].items():
        fmt = lambda v: f"{v:.1%}" if v is not None else "   n/a"  # noqa: E731
        print(
            f"  {name:<{width}}  {fmt(scores['non_equivalent_accuracy']):>7} "
            f"{fmt(scores['equivalent_accuracy']):>7} "
            f"{fmt(scores['balanced_accuracy']):>9}"
        )
    print("\n  (decision-only: none produces a witness, so all score 0 on H1's primary outcome)")

    out_dir = config.path_for("tables_dir")
    out_path = out_dir / "baselines.json"
    write_text(out_path, canonical_json(report) + "\n")
    print(f"\nreport: {out_path}")
    return 0 if check["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
