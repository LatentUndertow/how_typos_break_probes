"""Dataset wrappers used by the §6/§7 probe-training and evaluation pipeline.

9 datasets covering benign + 5 attack families:

- benign:       OpenOrca, Alpaca, Dolly15k, BitextCustomerSupport
- malicious:    DeepSet (direct PI), BIPIA + InjecAgent (indirect PI),
                HarmBench (jailbreak / harmful), AdvBench (harmful)
"""
from .base import Dataset
from .hf_dataset import HFDataset
from .openorca import OpenOrcaDataset
from .alpaca import AlpacaDataset
from .dolly15k import Dolly15kDataset
from .bitext_customer_support import BitextCustomerSupportDataset
from .deepset import DeepsetDataset
from .bipia import BIPIADataset
from .injecagent import InjecAgentDataset
from .harmbench import HarmBenchDataset
from .advbench import AdvBenchDataset

__all__ = [
    "Dataset",
    "HFDataset",
    "OpenOrcaDataset",
    "AlpacaDataset",
    "Dolly15kDataset",
    "BitextCustomerSupportDataset",
    "DeepsetDataset",
    "BIPIADataset",
    "InjecAgentDataset",
    "HarmBenchDataset",
    "AdvBenchDataset",
]
