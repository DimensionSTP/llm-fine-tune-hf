from typing import List, Protocol, Any

from vllm import LLM

from src.utils.reward_embedding import VllmEmbedding


class RewardManagerLike(Protocol):
    rewards: List[object]


def prepare_colocated_vllm_models(
    reward_manager: RewardManagerLike,
    generation_model: LLM,
) -> None:
    embeddings = _collect_reward_vllm_embeddings(reward_manager=reward_manager)
    if len(embeddings) == 0:
        return

    for embedding in embeddings:
        embedding.get_model()

    for embedding in embeddings:
        recapture_vllm_cuda_graphs(model=embedding.get_model())

    recapture_vllm_cuda_graphs(model=generation_model)


def recapture_vllm_cuda_graphs(
    model: LLM,
) -> None:
    model.collective_rpc(_recapture_worker_cuda_graphs)


def _collect_reward_vllm_embeddings(
    reward_manager: RewardManagerLike,
) -> List[VllmEmbedding]:
    embeddings = []
    seen_ids = set()
    for reward in reward_manager.rewards:
        embedding = getattr(reward, "embedding", None)
        if not isinstance(embedding, VllmEmbedding):
            continue
        embedding_id = id(embedding)
        if embedding_id in seen_ids:
            continue
        embeddings.append(embedding)
        seen_ids.add(embedding_id)
    return embeddings


def _recapture_worker_cuda_graphs(
    worker: Any,
) -> float:
    from vllm.v1.worker.workspace import unlock_workspace

    unlock_workspace()
    return worker.compile_or_warm_up_model()
