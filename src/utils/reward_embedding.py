from typing import Dict, List, Optional
import os

import numpy as np

from vllm import LLM


class VllmEmbedding:
    def __init__(
        self,
        model_path: str,
        model_base: str,
        model_ft_data: str,
        model_default_upload_user: str,
        num_gpus: int,
        seed: int,
        gpu_memory_utilization: float,
        max_length: int,
        instruction_length: int,
        instruction: str,
        device_id: Optional[int],
        master_addr: Optional[str],
        master_port: Optional[int],
        nccl_socket_ifname: Optional[str],
        nccl_ib_disable: Optional[int],
        preserved_env_keys: List[str],
        isolated_env_keys: List[str],
    ) -> None:
        if model_ft_data == "base":
            model_id = f"{model_default_upload_user}/{model_base}"
        else:
            model_id = os.path.join(
                model_path,
                model_base,
                model_ft_data,
            )

        self.model = None
        self.model_id = model_id
        self.tensor_parallel_size = 1 if device_id is not None else num_gpus
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_length = max_length + instruction_length
        self.instruction = instruction
        self.device_id = device_id
        self.master_addr = master_addr
        self.master_port = master_port
        self.nccl_socket_ifname = nccl_socket_ifname
        self.nccl_ib_disable = nccl_ib_disable
        self.preserved_env_keys = preserved_env_keys
        self.isolated_env_keys = isolated_env_keys

    def __call__(
        self,
        input_text: str,
        is_query: bool,
    ) -> np.ndarray:
        if is_query:
            input_text = self.get_detailed_instruction(query=input_text)
        embedding = self.embed(input_text=input_text)
        return embedding

    def embed(
        self,
        input_text: str,
    ) -> np.ndarray:
        model = self.get_model()
        output = model.embed(
            prompts=input_text,
            use_tqdm=False,
        )
        embedding = output[0].outputs.embedding
        embedding = np.array(
            embedding,
            dtype=np.float32,
        )
        return embedding

    def get_detailed_instruction(
        self,
        query: str,
    ) -> str:
        instruction = f"Instruct: {self.instruction}\nQuery:{query}"
        return instruction

    def get_model(self) -> LLM:
        if self.model is None:
            saved_env = self._capture_env()
            self._prepare_vllm_env()
            try:
                self.model = LLM(
                    model=self.model_id,
                    tensor_parallel_size=self.tensor_parallel_size,
                    seed=self.seed,
                    trust_remote_code=True,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_length,
                )
            finally:
                self._restore_env(saved_env=saved_env)
        return self.model

    def _capture_env(self) -> Dict[str, Optional[str]]:
        return {key: os.environ.get(key) for key in self.preserved_env_keys}

    def _prepare_vllm_env(self) -> None:
        if self.device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        if self.master_addr is not None:
            os.environ["MASTER_ADDR"] = str(self.master_addr)
        if self.master_port is not None:
            os.environ["MASTER_PORT"] = str(self.master_port)
        if self.nccl_socket_ifname is not None:
            os.environ["NCCL_SOCKET_IFNAME"] = str(self.nccl_socket_ifname)
        if self.nccl_ib_disable is not None:
            os.environ["NCCL_IB_DISABLE"] = str(self.nccl_ib_disable)

        os.environ.setdefault(
            "VLLM_WORKER_MULTIPROC_METHOD",
            "spawn",
        )
        for var in self.isolated_env_keys:
            if var in os.environ:
                del os.environ[var]

    def _restore_env(
        self,
        saved_env: Dict[str, Optional[str]],
    ) -> None:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
