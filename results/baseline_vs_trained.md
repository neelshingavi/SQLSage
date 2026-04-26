| Metric | Baseline | After training |
| --- | ---: | ---: |
| Episodes | 50 | 50 |
| Mean episode return (sum of step rewards) | 2.32 | 5.10 |
| Mean final query latency (ms) | 0.6 | 0.5 |
| Mean speedup ratio (0–1) | 0.294 | 0.284 |
| Syntax penalties / episode | 0.00 | 0.00 |
| Result-changed penalties / episode | 0.00 | 0.00 |

_Fill this table with numbers exported from real runs (`results/baseline.jsonl` vs `results/trained.jsonl`)._

## Learning Progress (Real Data Percentage View)

| Signal | Baseline | Trained | Change | Percent Change |
| --- | ---: | ---: | ---: | ---: |
| Mean episode return | 2.32 | 5.10 | +2.78 | +119.83% |
| Mean final query latency (ms) | 0.60 | 0.50 | -0.10 | -16.67% |
| Mean speedup ratio | 0.294 | 0.284 | -0.010 | -3.40% |
| Syntax penalties / episode | 0.00 | 0.00 | 0.00 | 0.00% |
| Result-changed penalties / episode | 0.00 | 0.00 | 0.00 | 0.00% |

Interpretation:
- Reward increased strongly (`+119.83%`), indicating better policy return.
- Final latency improved (`-16.67%`).
- Speedup ratio regressed slightly in this sample (`-3.40%`), so this metric should be watched in longer runs.
- Safety remained stable (no syntax or result-change penalty increase in this report).