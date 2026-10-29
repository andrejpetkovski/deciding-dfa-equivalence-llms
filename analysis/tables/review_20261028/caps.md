# Output-cap diagnostics and trace-prompt sensitivity

Post-review exploratory sensitivity; frozen primary estimates unchanged

## Definitions and coverage

finish_reason=length OR reported completion_tokens >= requested decoding.max_tokens; report agreement between the two signals

Primary sample 0 only: 2,400 calls from 200 DFA pairs. Requested cap: [12288]. There are 113 cap hits; the two cap signals disagree on 0 calls. Of 167 parse failures, 111 are empty capped answers, 2 are capped non-JSON answers, and 54 are uncapped non-JSON answers.

Grades are joined by run_id to the terminal raw attempt using the frozen grader's chronological selection. Identity, response, finish reason, and usage fields must agree. Final answers are extracted with the frozen adapter; no reasoning text is promoted into an answer and no response is reparsed or repaired.

## Cap hits and failure types

| Group | Calls | Capped | Cap rate | Empty capped | Non-JSON capped | Non-JSON uncapped | All parse failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| EQ_EXPAND | 600 | 38 | 6.33% | 37 | 1 | 15 | 53 |
| EQ_ISO | 600 | 32 | 5.33% | 31 | 1 | 13 | 45 |
| NEQ_DEPTH | 1200 | 43 | 3.58% | 43 | 0 | 26 | 69 |
| 1-2 | 300 | 1 | 0.33% | 1 | 0 | 4 | 5 |
| 3-4 | 300 | 2 | 0.67% | 2 | 0 | 8 | 10 |
| 5-8 | 300 | 13 | 4.33% | 13 | 0 | 8 | 21 |
| 9-12 | 300 | 27 | 9.00% | 27 | 0 | 6 | 33 |
| equivalent | 1200 | 70 | 5.83% | 68 | 2 | 28 | 98 |
| P0 | 1200 | 53 | 4.42% | 52 | 1 | 20 | 73 |
| P1 | 1200 | 60 | 5.00% | 59 | 1 | 34 | 94 |
| deepseek/deepseek-v4-flash-0731 | 800 | 19 | 2.38% | 19 | 0 | 0 | 19 |
| xiaomi/mimo-v2.5 | 800 | 80 | 10.00% | 79 | 1 | 54 | 134 |
| z-ai/glm-5.3-flash | 800 | 14 | 1.75% | 13 | 1 | 0 | 14 |

Exact-depth counts and model/prompt/family intersections are in `caps.json` and `cap_breakdowns.csv`, with both original and cap-excluded denominators.

## H1/H2 sensitivity

The same frozen GEE mean model is fitted after excluding capped calls. H1 uses exact oracle depth, the frozen prompt interaction, rendering and model indicators, and pair-cluster robust uncertainty. H2 averages P1-P0 predicted probability differences over the retained non-equivalent calls, with a pair-clustered percentile bootstrap. Each repeated draw of a pair is assigned a distinct bootstrap cluster ID. No new multiplicity-adjusted test is claimed.

H1 primary refit: -0.1860 log-odds/depth [-0.2374, -0.1347]. Without caps: -0.1329 [-0.1922, -0.0736].

H2 original point refit: -4.83 pp. Without caps: -4.96 pp [-8.14, -1.89]; 1157 retained non-equivalent calls from 100 pairs. Bootstrap: 2000/2000 fits used; seed 20261028. These intervals are exploratory; the original H2 interval and Holm correction remain as reported in the frozen-results artifacts.

| Population | Non-equivalent decision accuracy | Equivalent decision accuracy | Valid-witness success |
|---|---:|---:|---:|
| Primary | 1130/1200 (94.17%) | 1026/1200 (85.50%) | 1057/1200 (88.08%) |
| Cap-excluded | 1130/1157 (97.67%) | 1026/1130 (90.80%) | 1057/1157 (91.36%) |

## Trace-prompt decomposition on non-equivalent pairs

| Outcome | P0 | P1 | P1-P0 count |
|---|---:|---:|---:|
| empty_capped | 21 | 22 | +1 |
| false_equivalence | 0 | 1 | +1 |
| invalid_witness | 28 | 45 | +17 |
| longer_valid_witness | 40 | 27 | -13 |
| non_json_uncapped | 8 | 18 | +10 |
| shortest_valid_witness | 503 | 487 | -16 |

P1 traces were checked on 559 non-equivalent responses: 22 were incorrect; 23 invalid witnesses nevertheless had correct traces. Trace correctness is not required for the primary valid-witness metric. Trace errors overlap witness errors and are not another disjoint term in the failure table.

| Prompt | Non-equivalent calls | Median completion tokens | Median nonempty final-answer characters | Nonempty answers |
|---|---:|---:|---:|---:|
| P0 | 600 | 2585.0 | 40 | 579 |
| P1 | 600 | 2976.0 | 136.0 | 578 |

Character length is a directly measured description of visible answers, not a token count or a measure of reasoning. Completion usage includes provider accounting; its allocation to hidden reasoning versus final text is not identified here. Nonempty-answer summaries are conditional on an answer existing.

## Interpretation and limits

Cap exclusion selects completed calls using an observed outcome. It does not estimate performance under a larger output budget or isolate a reasoning mechanism. Non-cap parse failures still count as errors.

The cap-excluded depth coefficient remains negative, but is smaller in magnitude. The trace-prompt penalty remains negative. These observations show that the original patterns are not solely due to the observed capped calls; they do not identify how uncapped reasoning, formatting demands, or a larger budget would change outcomes. Correct traces of an invalid word do not certify a distinction.

## Reproduction

Run `python -m analysis.review_caps`. The original raw log, grades, configuration, and confirmatory modules are unchanged. `caps.json` contains all estimates, subgroup counts, input digests, and runtime versions; `cap_calls.csv` contains per-call diagnostics without response text.
