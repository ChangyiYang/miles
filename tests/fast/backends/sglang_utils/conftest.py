from __future__ import annotations

import functools
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

_FAKE_HF_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "vocab_size": 1000,
    "max_position_embeddings": 512,
    "torch_dtype": "bfloat16",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


@functools.cache
def make_fake_checkpoint() -> str:
    """A checkpoint directory that is only real enough for ServerArgs to resolve its config."""
    path = Path(tempfile.mkdtemp(prefix="miles-fake-checkpoint-"))
    (path / "config.json").write_text(json.dumps(_FAKE_HF_CONFIG))
    return str(path)


def make_engine_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field ``_compute_server_args`` touches."""
    defaults: dict[str, Any] = dict(
        hf_checkpoint=make_fake_checkpoint(),
        seed=42,
        offload_rollout=False,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=1,
        sglang_dp_size=1,
        sglang_pp_size=1,
        sglang_ep_size=1,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        fp16=False,
        lora_rank=0,
        sglang_api_key=None,
        lora_adapter_path=None,
        multi_lora=False,
        colocate=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)
