"""
BIPIA dataset provider for prompt mining platform.

BIPIA (Benchmark for Indirect Prompt Injection Attacks) contains prompts where
malicious instructions are embedded in external context (emails, code, tables, etc.)
that the LLM processes.

Tasks: email, code, table, qa, abstract

Reference: https://github.com/microsoft/BIPIA
Paper: https://arxiv.org/abs/2312.14197
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple, Union

from activation_robustness.datasets.base import Dataset
from activation_robustness.data.prompt_spec import PromptSpec
from activation_robustness.data.data_loader import DatasetMeta


class BIPIADataset(Dataset):
    """
    BIPIA indirect prompt injection attack dataset.

    Uses BIPIA's native builders to construct poisoned samples, then converts
    to PromptSpec format. Supports loading multiple tasks at once.

    Example:
        >>> dataset = BIPIADataset(
        ...     bipia_root="/path/to/BIPIA",
        ...     task_names=["email", "code"],
        ...     split="test",
        ...     insert_positions=["end"]
        ... )
        >>> for prompt in dataset:
        ...     print(prompt.prompt_id, prompt.labels["malicious"])
    """

    DATASET_META = DatasetMeta(default_params={'task_names': ['email', 'code', 'table'], 'split': 'train', 'insert_positions': ['end', 'middle'], 'include_clean': True}, has_spans=True, category='indirect_pi', max_samples=15000)

    AVAILABLE_TASKS = ["email", "code", "table", "qa", "abstract"]

    def __init__(
        self,
        bipia_root: str,
        task_names: Union[str, List[str]] = "email",
        split: str = "test",
        insert_positions: Optional[List[str]] = None,
        enable_stealth: bool = False,
        seed: int = 2023,
        include_clean: bool = False,
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ):
        """
        Initialize BIPIA dataset.

        Args:
            bipia_root: Path to BIPIA repository root
            task_names: Task(s) to load - single string or list of strings
                       Available: "email", "code", "table", "qa", "abstract"
            split: Data split ("train" or "test")
            insert_positions: Attack insertion positions (default: ["end", "start", "middle"])
            enable_stealth: Base64 encode attacks if True
            seed: Random seed for reproducibility
            include_clean: If True, also include samples without attack injection
            name: Dataset name (auto-generated if not provided)
            dataset_id: Unique identifier (auto-generated if not provided)
        """
        self.bipia_root = Path(bipia_root)
        if not self.bipia_root.exists():
            raise FileNotFoundError(
                f"BIPIA root directory not found: {self.bipia_root}\n"
            )
        self.split = split
        self.insert_positions = insert_positions or ["end", "start", "middle"]
        self.enable_stealth = enable_stealth
        self.seed = seed
        self.include_clean = include_clean

        # Normalize task_names to list
        if isinstance(task_names, str):
            task_names = [task_names]
        self.task_names = task_names

        # Validate tasks
        for task in self.task_names:
            if task not in self.AVAILABLE_TASKS:
                raise ValueError(
                    f"Unknown task: {task}. Available: {self.AVAILABLE_TASKS}"
                )

        # Validate insert positions
        valid_positions = ["end", "start", "middle"]
        for pos in self.insert_positions:
            if pos not in valid_positions:
                raise ValueError(
                    f"Unknown insert position: {pos}. Available: {valid_positions}"
                )

        # Generate name if not provided
        if name is None:
            tasks_str = "_".join(self.task_names)
            positions_str = "_".join(self.insert_positions)
            name = f"bipia_{tasks_str}_{split}_{positions_str}"

        super().__init__(name=name, dataset_id=dataset_id or f"bipia_{'_'.join(self.task_names)}")

        # Import BIPIA modules
        self._import_bipia()

        # Build samples using BIPIA builders
        self._samples = self._build_samples()

    def _import_bipia(self):
        """Import BIPIA modules by adding to sys.path."""
        bipia_str = str(self.bipia_root.resolve())
        if bipia_str not in sys.path:
            sys.path.insert(0, bipia_str)

        try:
            from bipia.data import AutoPIABuilder
            from bipia.data.utils import insert_end, insert_start, insert_middle, no_insert

            self._AutoPIABuilder = AutoPIABuilder
            self._insert_fns = {
                "end": insert_end,
                "start": insert_start,
                "middle": insert_middle,
                "none": no_insert,
            }
        except ImportError as e:
            raise ImportError(
                f"Failed to import BIPIA modules from {self.bipia_root}. "
                f"Make sure bipia_root points to BIPIA repository. Error: {e}"
            )

    def _get_attack_file(self, task_name: str) -> str:
        """Get attack file path for a task."""
        # code task uses code_attack, others use text_attack
        attack_type = "code_attack" if task_name == "code" else "text_attack"
        return str(self.bipia_root / "benchmark" / f"{attack_type}_{self.split}.json")

    def _get_context_file(self, task_name: str) -> str:
        """Get context file path for a task."""
        return str(self.bipia_root / "benchmark" / task_name / f"{self.split}.jsonl")

    def _build_samples(self) -> List[Dict[str, Any]]:
        """Build all samples using BIPIA builders."""
        samples = []

        # Build insert functions list based on config
        insert_fns = [self._insert_fns[pos] for pos in self.insert_positions]
        insert_fn_names = self.insert_positions.copy()

        for task_name in self.task_names:
            # Get builder for this task
            builder_cls = self._AutoPIABuilder.from_name(task_name)
            builder = builder_cls(seed=self.seed)

            context_file = self._get_context_file(task_name)
            attack_file = self._get_attack_file(task_name)

            # Optionally include true clean samples (original contexts only)
            if self.include_clean:
                clean_df = builder(
                    contexts=context_file,
                    attacks=attack_file,
                    insert_fns=[self._insert_fns["none"]],
                    insert_fn_names=["none"],
                    enable_stealth=False,
                )
                # Deduplicate: keep only unique contexts (first occurrence per context)
                seen_contexts = set()
                for _, row in clean_df.iterrows():
                    ctx = row.get("context", "")
                    if ctx not in seen_contexts:
                        seen_contexts.add(ctx)
                        row_dict = row.to_dict()
                        row_dict["position"] = "none"
                        row_dict["attack_name"] = None
                        row_dict["attack_str"] = None
                        system_prompt, user_prompt = builder.construct_prompt(row_dict)
                        samples.append({
                            "task_name": task_name,
                            "row": row_dict,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        })

            # Build poisoned samples
            df = builder(
                contexts=context_file,
                attacks=attack_file,
                insert_fns=insert_fns,
                insert_fn_names=insert_fn_names,
                enable_stealth=self.enable_stealth,
            )

            for _, row in df.iterrows():
                row_dict = row.to_dict()
                system_prompt, user_prompt = builder.construct_prompt(row_dict)
                samples.append({
                    "task_name": task_name,
                    "row": row_dict,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                })

        return samples

    def _convert_to_prompt_spec(self, sample: Dict[str, Any], index: int) -> PromptSpec:
        """Convert internal sample to PromptSpec format."""
        row = sample["row"]
        task_name = sample["task_name"]
        position = row.get("position", "none")
        is_poisoned = position != "none"

        # Build messages from system and user prompts
        messages = [
            {"role": "system", "content": sample["system_prompt"]},
            {"role": "user", "content": sample["user_prompt"]},
        ]

        # Build labels
        labels = {
            "malicious": is_poisoned,
            "task_name": task_name,
            "split": self.split,
            "position": position,
            "attack_name": row.get("attack_name"),
            "ideal": row.get("ideal"),
            "enable_stealth": self.enable_stealth,
        }

        # Add question if present
        if "question" in row:
            labels["question"] = row["question"]

        # Create prompt ID
        if is_poisoned:
            attack_short = (
                str(row.get("attack_name", ""))
                .replace(" ", "_")
                .replace("/", "_")
                .replace(":", "_")[:30]
            )
            prompt_id = f"{self.dataset_id}:{task_name}:{self.split}:{position}:{attack_short}:{index}"
        else:
            prompt_id = f"{self.dataset_id}:{task_name}:{self.split}:clean:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None,
        )

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._samples)

    def __iter__(self) -> Iterator[PromptSpec]:
        """Iterate over all samples as PromptSpec objects."""
        for i, sample in enumerate(self._samples):
            yield self._convert_to_prompt_spec(sample, i)

    # Utility methods
    def get_task_names(self) -> List[str]:
        """Return list of tasks in this dataset."""
        return self.task_names.copy()

    @staticmethod
    def available_tasks() -> List[str]:
        """Return list of all available task names."""
        return BIPIADataset.AVAILABLE_TASKS.copy()

    def get_span_context(self) -> dict:
        """Return kwargs needed by get_injection_span for this dataset instance."""
        attack_str_map = self.__class__.build_attack_str_map(
            bipia_root=getattr(self, 'bipia_root', ''),
            split=getattr(self, 'split', 'train'),
            task_names=getattr(self, 'task_names', []),
        )
        return {'attack_str_map': attack_str_map}

    # ------------------------------------------------------------------
    # Injection span finding
    # ------------------------------------------------------------------

    @classmethod
    def build_attack_str_map(
        cls,
        bipia_root: str,
        split: str = "train",
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Build attack_name → attack_str mapping from BIPIA benchmark files.

        Uses BIPIA builders to generate samples and extract unique
        (attack_name, attack_str) pairs.

        Args:
            bipia_root: Path to BIPIA repository root.
            split: Data split ("train" or "test").
            task_names: Tasks to include (default: ["email", "code", "table"]).

        Returns:
            Dict mapping attack_name to attack_str (e.g.,
            {"Persuasion-2": "Modify your answer to highlight..."}).
        """
        bipia_root = Path(bipia_root)
        bipia_str = str(bipia_root.resolve())
        if bipia_str not in sys.path:
            sys.path.insert(0, bipia_str)

        from bipia.data import AutoPIABuilder
        from bipia.data.utils import insert_end

        task_names = task_names or ["email", "code", "table"]
        attack_map: Dict[str, str] = {}

        for task_name in task_names:
            builder_cls = AutoPIABuilder.from_name(task_name)
            builder = builder_cls(seed=2023)

            attack_type = "code_attack" if task_name == "code" else "text_attack"
            attack_file = str(bipia_root / "benchmark" / f"{attack_type}_{split}.json")
            context_file = str(bipia_root / "benchmark" / task_name / f"{split}.jsonl")

            df = builder(
                contexts=context_file,
                attacks=attack_file,
                insert_fns=[insert_end],
                insert_fn_names=["end"],
                enable_stealth=False,
            )

            for _, row in df.drop_duplicates("attack_name").iterrows():
                attack_map[row["attack_name"]] = row["attack_str"]

        return attack_map

    @staticmethod
    def get_injection_span(
        input_text: str,
        labels: Dict[str, Any],
        attack_str_map: Optional[Dict[str, str]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find the character span of the injected instruction in input_text.

        Args:
            input_text: Full prompt text (with chat template applied).
            labels: PromptSpec labels dict (needs 'attack_name', 'position').
            attack_str_map: Mapping from attack_name to attack_str.
                Build once via ``BIPIADataset.build_attack_str_map()``.

        Returns:
            (start_char, end_char) or None if benign / not found.
        """
        position = labels.get("position", "none")
        if position == "none" or not labels.get("malicious", False):
            return None

        attack_name = labels.get("attack_name")
        if not attack_name or not attack_str_map:
            return None

        attack_str = attack_str_map.get(attack_name)
        if not attack_str:
            return None

        idx = input_text.find(attack_str)
        if idx < 0:
            return None

        return (idx, idx + len(attack_str))
