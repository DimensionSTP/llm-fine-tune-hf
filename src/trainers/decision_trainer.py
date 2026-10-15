from typing import Dict, List, Tuple, Optional, Any

import math

import numpy as np

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    EvalPrediction,
    ProcessorMixin,
    Trainer,
)

from ..datasets import build_decision_candidate_token_ids


class DecisionTrainer(Trainer):
    def __init__(
        self,
        candidate_markers: List[str],
        ece_num_bins: int,
        **kwargs: Any,
    ) -> None:
        if ece_num_bins <= 0:
            raise ValueError("decision.ece_num_bins must be positive.")
        self.candidate_markers = list(candidate_markers)
        self.ece_num_bins = ece_num_bins
        super().__init__(
            compute_metrics=self.compute_decision_metrics,
            **kwargs,
        )
        self.loss_is_scaled_for_ga = True
        self.model_accepts_loss_kwargs = False
        self.label_names = ["labels"]

    def train(
        self,
        **kwargs: Any,
    ) -> Any:
        checkpoint = kwargs.get("resume_from_checkpoint")
        if checkpoint:
            encoder_class = (
                AutoProcessor
                if isinstance(self.processing_class, ProcessorMixin)
                else AutoTokenizer
            )
            saved_encoder = encoder_class.from_pretrained(checkpoint)
            current_ids = build_decision_candidate_token_ids(
                data_encoder=self.processing_class,
                candidate_markers=self.candidate_markers,
            )
            saved_ids = build_decision_candidate_token_ids(
                data_encoder=saved_encoder,
                candidate_markers=self.candidate_markers,
            )
            if (
                current_ids != saved_ids
                or saved_encoder.chat_template != self.processing_class.chat_template
            ):
                raise ValueError(
                    "Decision resume requires the saved marker mapping and chat template."
                )
        return super().train(**kwargs)

    def get_eval_dataloader(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> DataLoader:
        even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = True
        try:
            return super().get_eval_dataloader(
                *args,
                **kwargs,
            )
        finally:
            self.accelerator.even_batches = even_batches

    def _prepare_inputs(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepared = super()._prepare_inputs(
            {key: value for key, value in inputs.items() if key != "labels"},
        )
        prepared["labels"] = inputs["labels"].to(
            device=self.args.device,
            dtype=torch.float32,
        )
        return prepared

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        scores = _compute_candidate_scores(
            model=model,
            inputs=inputs,
        )
        losses = _compute_question_losses(
            scores=scores,
            mask=inputs["candidate_mask"],
            labels=inputs["labels"],
        )
        count = kwargs.get("num_items_in_batch")
        loss = losses.mean() if count is None else losses.sum() / count
        if count is not None and self.args.average_tokens_across_devices:
            loss *= self.accelerator.num_processes // self.get_tp_size()
        if kwargs.get("return_outputs", False):
            return (
                loss,
                {"logits": scores},
            )
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]],
    ) -> Tuple[
        torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Optional[torch.Tensor]
    ]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(
                model=model,
                inputs=inputs,
                return_outputs=True,
            )
        if prediction_loss_only:
            return (
                loss.detach(),
                None,
                None,
            )
        metadata = inputs["decision_metadata"]
        device = outputs["logits"].device
        levels = torch.zeros_like(inputs["labels"])
        for index, row in enumerate(metadata):
            if row["task_type"] == "score":
                levels[index, : len(row["score_levels"])] = torch.tensor(
                    row["score_levels"],
                    device=device,
                )
        predictions = (
            outputs["logits"].detach(),
            inputs["candidate_mask"].to(dtype=torch.long),
            torch.tensor(
                [[row["hard_label"]] for row in metadata],
                device=device,
            ),
            torch.tensor(
                [
                    [
                        (
                            row["candidate_ids"].index("true")
                            if row["task_type"] == "noul"
                            else -1
                        )
                    ]
                    for row in metadata
                ],
                device=device,
            ),
            levels,
            torch.tensor(
                [[row["task_type"] == "score"] for row in metadata],
                device=device,
            ),
        )
        return (
            loss.detach(),
            predictions,
            inputs["labels"].detach(),
        )

    def _get_num_items_in_batch(
        self,
        batch_samples: List[Dict[str, Any]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not batch_samples:
            return None
        count = torch.tensor(
            sum(batch["labels"].shape[0] for batch in batch_samples),
            device=device,
        )
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            count = self.accelerator.gather(count).sum()
        count = count // (self.get_tp_size() * self.get_cp_size() * self.get_sp_size())
        return count

    def compute_decision_metrics(
        self,
        evaluation: EvalPrediction,
    ) -> Dict[str, float]:
        scores, mask, hard_labels, true_indices, levels, score_mask = (
            evaluation.predictions
        )
        probabilities = torch.softmax(
            torch.tensor(scores).masked_fill(
                torch.tensor(mask) <= 0,
                -torch.inf,
            ),
            dim=-1,
        ).numpy()
        return _compute_probability_metrics(
            probabilities=probabilities,
            labels=np.maximum(
                evaluation.label_ids,
                0,
            ),
            hard_labels=np.asarray(hard_labels).reshape(-1),
            ece_num_bins=self.ece_num_bins,
            true_indices=np.asarray(true_indices).reshape(-1),
            score_levels=np.where(
                np.asarray(mask) > 0,
                levels,
                0,
            ),
            score_mask=np.asarray(score_mask).reshape(-1) > 0,
        )


def predict_decisions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    inference_temperature: float,
) -> List[Dict[str, Any]]:
    if not math.isfinite(inference_temperature) or inference_temperature <= 0:
        raise ValueError("decision.inference_temperature must be finite and positive.")
    model.eval()
    predictions = []
    for batch in test_loader:
        inputs = {
            key: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            scores = _compute_candidate_scores(
                model=model,
                inputs=inputs,
            )
            probabilities = torch.softmax(
                scores / inference_temperature,
                dim=-1,
            )
        for index, metadata in enumerate(inputs["decision_metadata"]):
            count = len(metadata["candidate_ids"])
            predictions.append(
                {
                    **metadata,
                    "probabilities": probabilities[index, :count].cpu().tolist(),
                    "target_probabilities": inputs["labels"][index, :count]
                    .cpu()
                    .tolist(),
                }
            )
    return predictions


def collect_decision_results(
    predictions: List[Dict[str, Any]],
    ece_num_bins: int,
) -> Dict[str, Any]:
    by_index = {}
    for prediction in predictions:
        by_index.setdefault(
            prediction["row_index"],
            prediction,
        )
    ordered = [by_index[index] for index in sorted(by_index)]
    results = []
    for prediction in ordered:
        probabilities = prediction["probabilities"]
        ids = prediction["candidate_ids"]
        selected = int(np.argmax(probabilities))
        result = {
            "id": prediction["id"],
            "task_type": prediction["task_type"],
            "selected_id": ids[selected],
            "probabilities": dict(
                zip(
                    ids,
                    probabilities,
                )
            ),
            "confidence": probabilities[selected],
        }
        if prediction["has_target"]:
            result["target"] = (
                {"label_id": ids[prediction["hard_label"]]}
                if prediction["hard_label"] >= 0
                else {
                    "probabilities": dict(
                        zip(
                            ids,
                            prediction["target_probabilities"],
                        )
                    ),
                }
            )
        if prediction["task_type"] == "noul":
            result["true_probability"] = probabilities[ids.index("true")]
        elif prediction["task_type"] == "score":
            levels = prediction["score_levels"]
            result["selected_level"] = levels[selected]
            result["expected_score"] = sum(
                probability * level for probability, level in zip(probabilities, levels)
            )
        results.append(result)
    labelled = [prediction for prediction in ordered if prediction["has_target"]]
    metrics = {}
    if labelled:
        capacity = max(len(row["probabilities"]) for row in labelled)
        probabilities = np.zeros(
            (
                len(labelled),
                capacity,
            )
        )
        labels = np.zeros_like(probabilities)
        levels = np.zeros_like(probabilities)
        for index, row in enumerate(labelled):
            probabilities[index, : len(row["probabilities"])] = row["probabilities"]
            labels[index, : len(row["target_probabilities"])] = row[
                "target_probabilities"
            ]
            if row["task_type"] == "score":
                levels[index, : len(row["score_levels"])] = row["score_levels"]
        metrics = _compute_probability_metrics(
            probabilities=probabilities,
            labels=labels,
            hard_labels=np.asarray([row["hard_label"] for row in labelled]),
            ece_num_bins=ece_num_bins,
            true_indices=np.asarray(
                [
                    (
                        row["candidate_ids"].index("true")
                        if row["task_type"] == "noul"
                        else -1
                    )
                    for row in labelled
                ]
            ),
            score_levels=levels,
            score_mask=np.asarray([row["task_type"] == "score" for row in labelled]),
        )
    return {
        "results": results,
        "metrics": metrics,
    }


def _compute_candidate_scores(
    model: torch.nn.Module,
    inputs: Dict[str, Any],
) -> torch.Tensor:
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key
        not in {"candidate_token_ids", "candidate_mask", "labels", "decision_metadata"}
    }
    outputs = model(**model_inputs)
    attention = inputs["attention_mask"]
    positions = torch.arange(
        attention.shape[1],
        device=attention.device,
    ).expand_as(attention)
    last_positions = (
        positions.masked_fill(
            attention == 0,
            -1,
        )
        .max(dim=1)
        .values
    )
    if (last_positions < 0).any():
        raise ValueError("Decision input must contain at least one real token.")
    logits = outputs.logits[
        torch.arange(
            attention.shape[0],
            device=attention.device,
        ),
        last_positions,
    ].float()
    scores = logits.gather(
        dim=1,
        index=inputs["candidate_token_ids"],
    )
    scores = scores.masked_fill(
        ~inputs["candidate_mask"],
        -torch.inf,
    )
    if not torch.isfinite(scores[inputs["candidate_mask"]]).all():
        raise ValueError("Decision model returned non-finite candidate scores.")
    return scores


def _compute_question_losses(
    scores: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    log_probabilities = torch.log_softmax(
        scores,
        dim=-1,
    ).masked_fill(
        ~mask,
        0,
    )
    categorical = -(labels * log_probabilities).sum(dim=-1)
    binary_rows = mask.sum(dim=-1) == 2
    if binary_rows.any():
        binary = torch.nn.functional.binary_cross_entropy_with_logits(
            scores[binary_rows, 1] - scores[binary_rows, 0],
            labels[binary_rows, 1],
            reduction="none",
        )
        categorical = categorical.clone()
        categorical[binary_rows] = binary
    return categorical


def _compute_probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    hard_labels: np.ndarray,
    ece_num_bins: int,
    true_indices: np.ndarray,
    score_levels: np.ndarray,
    score_mask: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "num_examples": len(labels),
        "num_hard_labels": int((hard_labels >= 0).sum()),
        "nll": float(
            -(
                labels
                * np.log(
                    np.clip(
                        probabilities,
                        1e-30,
                        1,
                    )
                )
            )
            .sum(axis=1)
            .mean()
        ),
        "brier": float(np.square(probabilities - labels).sum(axis=1).mean()),
    }
    hard = hard_labels >= 0
    if hard.any():
        confidence = probabilities[hard].max(axis=1)
        correct = probabilities[hard].argmax(axis=1) == hard_labels[hard]
        metrics["accuracy"] = float(correct.mean())
        ece = 0.0
        for index in range(ece_num_bins):
            selected = (confidence >= index / ece_num_bins) & (
                confidence < (index + 1) / ece_num_bins
                if index < ece_num_bins - 1
                else confidence <= 1
            )
            if selected.any():
                ece += float(
                    selected.mean()
                    * abs(correct[selected].mean() - confidence[selected].mean())
                )
        metrics["ece"] = ece
    noul_mask = true_indices >= 0
    if noul_mask.any():
        metrics["noul_num_examples"] = int(noul_mask.sum())
        metrics["noul_true_probability"] = float(
            probabilities[noul_mask, true_indices[noul_mask]].mean()
        )
    if score_mask.any():
        expected = (probabilities[score_mask] * score_levels[score_mask]).sum(axis=1)
        target = (labels[score_mask] * score_levels[score_mask]).sum(axis=1)
        metrics["score_num_examples"] = int(score_mask.sum())
        metrics["score_expected_mae"] = float(np.abs(expected - target).mean())
    return metrics
