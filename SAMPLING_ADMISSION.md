# Generation sampling admission

This fork fixes the hard-coded `top_p=0.5` threshold in the original
`tommyip/mamba2-minimal` generator. Its base revision is
`4f4413731f24303b4e046962bce6ca5b6a3b13d6`;
`mamba2.py` had Git blob SHA `7c768e6e6a62447949863ada8544c4c78efd2dde`.
The repository retains its existing Apache-2.0 licence.

Generation admission rejects invalid `top_p`, `top_k`, temperature,
generation length, prompt token IDs and non-finite sampling logits. It
bounds requests to at most 4096 generated tokens and 8192 prompt tokens.
Resource scheduling, process liveness and execution authorization
remain outside this source-level correction.

When PyTorch and einops are already installed, run the CPU-only tests
from the repository root with:

    python -m unittest discover -s tests -p 'test_sampling_admission.py' -v

Tests use synthetic logits and weights, without downloading any model
or connecting to a runtime. They are not deployment or model-quality evidence.
