"""
Databricks Dolly 15k dataset from HuggingFace.

Single-turn instructions with optional context and category labels.
"""

from typing import Dict, Any, Optional

from activation_robustness.datasets.hf_dataset import HFDataset
from activation_robustness.data.prompt_spec import PromptSpec
from activation_robustness.data.data_loader import DatasetMeta


class Dolly15kDataset(HFDataset):
    """
    Databricks Dolly 15k dataset.

    Dataset: databricks/databricks-dolly-15k

    Schema (observed):
    - instruction: str
    - context: str
    - response: str
    - category: str
    """

    DATASET_META = DatasetMeta(default_params={'split': 'train', 'include_context': True, 'store_reference_response': False}, category='benign')

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        include_context: bool = True,
        store_reference_response: bool = True,
        **kwargs,
    ):
        self.include_context = include_context
        self.store_reference_response = store_reference_response
        super().__init__(
            repo_id="databricks/databricks-dolly-15k",
            split=split,
            name=name or "dolly_15k",
            dataset_id=dataset_id or "dolly_15k",
            **kwargs,
        )

    def _convert_to_prompt_spec(self, example: Dict[str, Any], index: int) -> PromptSpec:
        instruction = example.get("instruction", "") or ""
        context = example.get("context", "") or ""
        response = example.get("response", "") or ""
        category = example.get("category", None)

        if self.include_context and context.strip():
            content = f"{instruction}\n\n{context}"
        else:
            content = instruction

        messages = [{"role": "user", "content": content}]

        labels = {
            "malicious": False,
            "category": category,
            "has_context": bool(context.strip()),
        }
        if self.store_reference_response:
            labels["reference_response"] = response

        prompt_id = f"{self.dataset_id}:{self.split}:{index}"
        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None,
        )


