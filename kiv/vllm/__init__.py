"""KIV integration for vLLM via KV Connector V1 plugin API.

EXPERIMENTAL: This integration has not been tested or validated against a
running vLLM instance. The connector API, tensor layouts, and attention hook
may need adaptation for specific vLLM versions. Use at your own risk.
"""

from .connector import KIVConnector
from .topology import detect_topology_from_vllm
from .attention_hook import install_attention_hook, uninstall_attention_hook

__all__ = [
    "KIVConnector",
    "detect_topology_from_vllm",
    "install_attention_hook",
    "uninstall_attention_hook",
]
