from typing import List, Optional, Callable, Any

from .base import BaseReward


class RewardManager:
    def __init__(
        self,
        rewards: List[BaseReward],
    ) -> None:
        self.rewards = [reward for reward in rewards if reward.weight > 0]

    def get_reward_funcs(self) -> List[Callable]:
        return [RewardFunctionAdapter(reward=reward) for reward in self.rewards]


class NamespacedLogger:
    def __init__(
        self,
        reward_name: str,
        callback: Callable[[str, Any], None],
    ) -> None:
        self.reward_name = reward_name
        self.callback = callback

    def __call__(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.callback(
            f"{self.reward_name}/{key}",
            value,
        )


class RewardFunctionAdapter:
    def __init__(
        self,
        reward: BaseReward,
    ) -> None:
        self.reward = reward
        self.__name__ = reward.name

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> List[Optional[float]]:
        patched_kwargs = dict(kwargs)
        log_extra = patched_kwargs.get("log_extra")
        log_metric = patched_kwargs.get("log_metric")

        if callable(log_extra):
            patched_kwargs["log_extra"] = NamespacedLogger(
                reward_name=self.reward.name,
                callback=log_extra,
            )

        if callable(log_metric):
            patched_kwargs["log_metric"] = NamespacedLogger(
                reward_name=self.reward.name,
                callback=log_metric,
            )

        rewards = self.reward(
            *args,
            **patched_kwargs,
        )
        self.log_reward_outputs(
            rewards=rewards,
            log_extra=log_extra if callable(log_extra) else None,
            log_metric=log_metric if callable(log_metric) else None,
        )
        return rewards

    def log_reward_outputs(
        self,
        rewards: List[Optional[float]],
        log_extra: Optional[Callable[[str, Any], None]],
        log_metric: Optional[Callable[[str, Any], None]],
    ) -> None:
        if log_extra is not None:
            log_extra(
                f"{self.reward.name}/reward",
                rewards,
            )

        if log_metric is None:
            return

        total_count = len(rewards)
        if total_count == 0:
            return

        valid_rewards = [reward for reward in rewards if reward is not None]
        log_metric(
            f"{self.reward.name}/coverage",
            len(valid_rewards) / total_count,
        )
        if len(valid_rewards) == 0:
            return

        mean_reward = sum(valid_rewards) / len(valid_rewards)
        log_metric(
            f"{self.reward.name}/mean",
            mean_reward,
        )
