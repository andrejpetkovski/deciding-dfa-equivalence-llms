# Configuration-specific depth analysis

Exploratory primary-sample analysis: 100 NEQ pairs, 400 calls per model. All parse failures remain errors. Each model uses exact centered depth, prompt, rendering, and depth-by-prompt interaction, with exchangeable GEE and pair-cluster robust 95% intervals. Slopes average across the centered prompt coding. No multiplicity-adjusted heterogeneity test is claimed.

| Configuration | Depth slope | 95% CI |
|---|---:|---:|
| deepseek/deepseek-v4-flash-0731 | -0.1376 | [-0.2751, -0.0002] |
| xiaomi/mimo-v2.5 | -0.2357 | [-0.3028, -0.1687] |
| z-ai/glm-5.3-flash | -0.1374 | [-0.2359, -0.0389] |

| Configuration | Prompt | Depth bin | Successes / calls |
|---|---|---|---:|
| deepseek/deepseek-v4-flash-0731 | P0 | 1-2 | 50/50 |
| deepseek/deepseek-v4-flash-0731 | P0 | 3-4 | 48/50 |
| deepseek/deepseek-v4-flash-0731 | P0 | 5-8 | 50/50 |
| deepseek/deepseek-v4-flash-0731 | P0 | 9-12 | 46/50 |
| deepseek/deepseek-v4-flash-0731 | P1 | 1-2 | 50/50 |
| deepseek/deepseek-v4-flash-0731 | P1 | 3-4 | 45/50 |
| deepseek/deepseek-v4-flash-0731 | P1 | 5-8 | 48/50 |
| deepseek/deepseek-v4-flash-0731 | P1 | 9-12 | 46/50 |
| xiaomi/mimo-v2.5 | P0 | 1-2 | 47/50 |
| xiaomi/mimo-v2.5 | P0 | 3-4 | 46/50 |
| xiaomi/mimo-v2.5 | P0 | 5-8 | 38/50 |
| xiaomi/mimo-v2.5 | P0 | 9-12 | 35/50 |
| xiaomi/mimo-v2.5 | P1 | 1-2 | 48/50 |
| xiaomi/mimo-v2.5 | P1 | 3-4 | 41/50 |
| xiaomi/mimo-v2.5 | P1 | 5-8 | 34/50 |
| xiaomi/mimo-v2.5 | P1 | 9-12 | 30/50 |
| z-ai/glm-5.3-flash | P0 | 1-2 | 47/50 |
| z-ai/glm-5.3-flash | P0 | 3-4 | 46/50 |
| z-ai/glm-5.3-flash | P0 | 5-8 | 48/50 |
| z-ai/glm-5.3-flash | P0 | 9-12 | 42/50 |
| z-ai/glm-5.3-flash | P1 | 1-2 | 49/50 |
| z-ai/glm-5.3-flash | P1 | 3-4 | 44/50 |
| z-ai/glm-5.3-flash | P1 | 5-8 | 36/50 |
| z-ai/glm-5.3-flash | P1 | 9-12 | 43/50 |

## Categorical-depth sensitivity

The pooled alternative replaces the linear depth term and its prompt interaction with three depth-bin indicators and their prompt interactions, retaining model/rendering controls and pair clustering. Reference bin: 1-2. This relaxes linearity across bins but does not identify a within-bin slope.

| Bin contrast versus 1-2 | Log-odds contrast | 95% CI |
|---|---:|---:|
| bin[3-4] | -1.3302 | [-2.1327, -0.5277] |
| bin[5-8] | -1.8023 | [-2.5778, -1.0267] |
| bin[9-12] | -2.1879 | [-2.9057, -1.4701] |

Unequal bin widths do not imply midpoint coding: the original regression uses exact oracle depth minus 5.5. Its linear log-odds assumption is a summary constraint, not a proven law. Configuration-specific intervals are descriptive; differences in their significance are not a test of differences between models. Model and provider remain confounded. Full coefficients are in depth.json.

Reproduce with `python -m analysis.review_depth`.
