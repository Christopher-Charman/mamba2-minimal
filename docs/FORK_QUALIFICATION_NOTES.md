# Fork qualification notes — minimal Mamba-2

**Scope:** public PyTorch research fork. No production inference cutover is asserted.

## Upstream baseline
This repository is a minimal Mamba-2 / state-space duality implementation. Preserve upstream author attribution, Apache-2.0 license and NOTICE obligations where applicable. The documented CPU/MPS reference behavior is a scoped research claim, not general accuracy/throughput parity across providers.

```mermaid
flowchart LR
  W["Pinned model/checkpoint inputs"] --> S["SSM / Mamba-2 forward implementation"]
  S --> P["Sampling / logits path"]
  P --> T["CPU and device regressions"]
  T --> E["Acceptance / portability assessment"]
```

## Fork-specific evaluation
During October 2026 the owner maintained source-level sampling and CPU regression work in a dedicated PR. Checks on one reviewed head reported successful unit/CPU jobs, while an independent reviewer identity gate remained unconfigured. A successful code test, completed sampling experiment and independently accepted release are three different states.

## Qualification
Use pinned PyTorch/einops/transformers versions; test deterministic reference logits, CPU reproducibility, causal continuity, sampler inputs/bounds, nonfinite or degenerate cases and optional device backends independently. Record exact commit, test commands and numerical tolerance. Never infer local deployment or model-router integration from a merged fork commit.

**Publication:** keep original Apache 2.0 license and notices; changes are for review/qualification unless separately accepted for use.
