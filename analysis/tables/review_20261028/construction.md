# Construction and family/rendering audit

Audited 100 non-equivalent pairs. Every pair has a rooted transition bijection with exactly two acceptance mismatches.

- identical_stored_tables: 0/100
- R0_all_corresponding_labels_equal: 0/100
- R0_identical_displayed_transitions: 0/100
- R1_all_corresponding_labels_equal: 0/100
- R1_identical_displayed_transitions: 0/100

R0 uses stored order after independent A/B permutations in src/generate.py. R1 additionally uses separate random streams keyed by pair seed, rendering, and A/B name in src/render_prompts.py. Shared label vocabulary does not imply shared state correspondence. A transition alignment still exposes the two acceptance changes; this construction does not cover general NEQ structure.

All rates below retain parse failures as errors and use primary sample 0 only. They are descriptive; they do not establish a causal rendering effect.

| Family | Rendering | Metric | Successes / calls | Rate |
|---|---|---|---:|---:|
| EQ_EXPAND | R0 | decision_accuracy | 252/300 | 84.00% |
| EQ_EXPAND | R1 | decision_accuracy | 250/300 | 83.33% |
| EQ_ISO | R0 | decision_accuracy | 268/300 | 89.33% |
| EQ_ISO | R1 | decision_accuracy | 256/300 | 85.33% |
| NEQ_DEPTH | R0 | valid_witness | 536/600 | 89.33% |
| NEQ_DEPTH | R1 | valid_witness | 521/600 | 86.83% |

Model/prompt intersections and per-pair structural checks are in construction.json. Reproduce with `python -m analysis.review_construction`.

Different-structure non-equivalent pairs would require a separate experiment. A useful extension is an acceptance change in one member of an EQ_EXPAND pair, followed by oracle verification, acceptance-count controls, and depth matching. No such experiment was run, and its outcomes are not claimed here.
