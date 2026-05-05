"""
Bitext Customer Support dataset from HuggingFace.

Benign customer service conversations for LLM chatbot training.
"""

import random
import re
from typing import Dict, Any, Optional
from activation_robustness.datasets.hf_dataset import HFDataset
from activation_robustness.data.prompt_spec import PromptSpec
from activation_robustness.data.data_loader import DatasetMeta


class BitextCustomerSupportDataset(HFDataset):
    """
    Bitext Customer Support LLM Chatbot Training Dataset.

    Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset

    Customer service conversations with intent classification.
    All examples are benign business communications.

    Schema:
    - flags: str (quality/metadata flags)
    - instruction: str (customer request/question)
    - category: str (11 categories, e.g., ORDER, ACCOUNT)
    - intent: str (27 intents, e.g., cancel_order, track_order)
    - response: str (support agent response)

    Splits:
    - train: 26,900 rows
    """

    DATASET_META = DatasetMeta(default_params={'split': 'train'}, category='benign')

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            repo_id="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split=split,
            name=name or "bitext_customer_support",
            dataset_id=dataset_id or "bitext_customer_support",
            **kwargs
        )

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        instruction = example.get("instruction", "")
        category = example.get("category", "")
        intent = example.get("intent", "")

        # Replace {{Order Number}} placeholder with a random order number
        rng = random.Random(index)
        instruction = re.sub(
            r"\{\{Order Number\}\}",
            f"ORD-{rng.randint(100000, 999999)}",
            instruction,
        )

        messages = [
            {"role": "user", "content": instruction}
        ]

        labels = {
            "malicious": False,
            "category": category,
            "intent": intent,
        }

        prompt_id = f"{self.dataset_id}:{self.split}:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None
        )
