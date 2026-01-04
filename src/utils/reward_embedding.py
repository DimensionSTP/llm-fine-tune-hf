from typing import Optional
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
        instruction: str,
        device_id: Optional[int],
        master_addr: Optional[str],
        master_port: Optional[int],
        nccl_socket_ifname: Optional[str],
        nccl_ib_disable: Optional[int],
    ) -> None:
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if master_addr is not None:
            os.environ["MASTER_ADDR"] = str(master_addr)
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)
        if nccl_socket_ifname is not None:
            os.environ["NCCL_SOCKET_IFNAME"] = str(nccl_socket_ifname)
        if nccl_ib_disable is not None:
            os.environ["NCCL_IB_DISABLE"] = str(nccl_ib_disable)

        os.environ.setdefault(
            "VLLM_WORKER_MULTIPROC_METHOD",
            "spawn",
        )
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            if var in os.environ:
                del os.environ[var]

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
        self.max_length = max_length
        self.instruction = instruction

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
        model = self._get_model()
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

    def _get_model(self) -> LLM:
        if self.model is None:
            self.model = LLM(
                model=self.model_id,
                task="embed",
                tensor_parallel_size=self.tensor_parallel_size,
                seed=self.seed,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_length,
            )
        return self.model
