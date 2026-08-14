from typing import Dict, List, Tuple, Optional, Callable, Any
import os
from functools import partial
import json
import statistics
import subprocess
import sys
import tempfile
import time

import torch
from torch import nn
import torch.nn.functional as F


def run_vision_patch_embedding_probe(
    signature: Dict[str, Any],
    probe_config: Dict[str, Any],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="vision_patch_embedding_probe_",
    ) as probe_dir:
        ready_path = os.path.join(
            probe_dir,
            "ready.json",
        )
        result_path = os.path.join(
            probe_dir,
            "result.json",
        )
        state_path = os.path.join(
            probe_dir,
            "state.json",
        )
        stderr_path = os.path.join(
            probe_dir,
            "stderr.log",
        )
        payload = {
            "signature": signature,
            "probe_config": probe_config,
            "ready_path": ready_path,
            "result_path": result_path,
            "state_path": state_path,
        }
        command = [
            sys.executable,
            os.path.abspath(__file__),
            json.dumps(payload),
        ]
        with open(
            stderr_path,
            "w",
            encoding="utf-8",
        ) as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                text=True,
            )
            startup_result = _wait_for_probe_startup(
                process=process,
                ready_path=ready_path,
                state_path=state_path,
                stderr_path=stderr_path,
                timeout_seconds=float(probe_config["startup_timeout_seconds"]),
            )
            if startup_result is not None:
                return _resolve_probe_decision(
                    result=startup_result,
                    probe_config=probe_config,
                )

            operation_result = _wait_for_probe_result(
                process=process,
                result_path=result_path,
                state_path=state_path,
                stderr_path=stderr_path,
                timeout_seconds=float(probe_config["operation_timeout_seconds"]),
            )
        return _resolve_probe_decision(
            result=operation_result,
            probe_config=probe_config,
        )


def _wait_for_probe_startup(
    process: subprocess.Popen,
    ready_path: str,
    state_path: str,
    stderr_path: str,
    timeout_seconds: float,
) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if os.path.isfile(ready_path):
            return None
        return_code = process.poll()
        if return_code is not None:
            return _build_process_failure(
                phase="startup",
                return_code=return_code,
                stderr=_read_text(path=stderr_path),
                state_path=state_path,
            )
        time.sleep(0.05)

    process.kill()
    process.wait()
    return {
        "status": "startup_timeout",
        "phase": "startup",
        "return_code": process.returncode,
        "stderr": _read_text(path=stderr_path),
    }


def _wait_for_probe_result(
    process: subprocess.Popen,
    result_path: str,
    state_path: str,
    stderr_path: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if os.path.isfile(result_path):
            result = _read_json(path=result_path)
            process.wait()
            result["return_code"] = process.returncode
            result["stderr"] = _read_text(path=stderr_path)
            return result
        return_code = process.poll()
        if return_code is not None:
            if os.path.isfile(result_path):
                result = _read_json(path=result_path)
                result["return_code"] = return_code
                result["stderr"] = _read_text(path=stderr_path)
                return result
            return _build_process_failure(
                phase="operation",
                return_code=return_code,
                stderr=_read_text(path=stderr_path),
                state_path=state_path,
            )
        time.sleep(0.05)

    process.kill()
    process.wait()
    return {
        "status": "operation_timeout",
        "phase": _read_probe_phase(state_path=state_path),
        "return_code": process.returncode,
        "stderr": _read_text(path=stderr_path),
    }


def _build_process_failure(
    phase: str,
    return_code: int,
    stderr: str,
    state_path: str,
) -> Dict[str, Any]:
    stderr_lower = stderr.lower()
    if "out of memory" in stderr_lower:
        status = "oom"
    elif return_code < 0:
        status = "signal_termination"
    else:
        status = "child_failure"
    return {
        "status": status,
        "phase": (
            _read_probe_phase(state_path=state_path) if phase == "operation" else phase
        ),
        "return_code": return_code,
        "signal": -return_code if return_code < 0 else None,
        "stderr": stderr,
    }


def _resolve_probe_decision(
    result: Dict[str, Any],
    probe_config: Dict[str, Any],
) -> Dict[str, Any]:
    status = result["status"]
    if status != "completed":
        native_failure = result["phase"] in {
            "native_correctness",
            "native_benchmark",
        }
        if status in {"signal_termination", "operation_timeout"} and native_failure:
            result["decision"] = "linear"
            result["decision_reason"] = status
            return result
        result["decision"] = "error"
        result["decision_reason"] = status
        return result

    measurements = result["measurements"]
    maximum_ratio = max(measurement["slowdown_ratio"] for measurement in measurements)
    maximum_difference = max(
        measurement["native_milliseconds"] - measurement["linear_milliseconds"]
        for measurement in measurements
    )
    use_linear = maximum_ratio >= float(probe_config["slowdown_ratio"])
    result["decision"] = "linear" if use_linear else "native"
    result["decision_reason"] = (
        "performance_threshold_matched"
        if use_linear
        else "performance_threshold_not_matched"
    )
    result["maximum_slowdown_ratio"] = maximum_ratio
    result["maximum_slowdown_milliseconds"] = maximum_difference
    return result


def _run_probe_child(
    payload: Dict[str, Any],
) -> None:
    signature = payload["signature"]
    probe_config = payload["probe_config"]
    device = _resolve_probe_device()
    _write_json(
        path=payload["ready_path"],
        payload={
            "status": "ready",
            "device": str(device),
        },
    )
    fp32_correctness = _verify_equivalence(
        signature=signature,
        device=device,
        state_path=payload["state_path"],
        dtype=torch.float32,
        atol=float(probe_config["fp32_equivalence_atol"]),
        rtol=float(probe_config["fp32_equivalence_rtol"]),
    )
    runtime_dtype = _resolve_torch_dtype(dtype_name=signature["dtype"])
    runtime_correctness = (
        fp32_correctness
        if runtime_dtype == torch.float32
        else _verify_equivalence(
            signature=signature,
            device=device,
            state_path=payload["state_path"],
            dtype=runtime_dtype,
            atol=float(probe_config["runtime_equivalence_atol"]),
            rtol=float(probe_config["runtime_equivalence_rtol"]),
        )
    )
    measurements = _measure_operators(
        signature=signature,
        probe_config=probe_config,
        device=device,
        state_path=payload["state_path"],
    )
    _write_json(
        path=payload["result_path"],
        payload={
            "status": "completed",
            "correctness": {
                "float32": fp32_correctness,
                "runtime_dtype": runtime_correctness,
            },
            "measurements": measurements,
        },
    )


def _resolve_probe_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Vision patch embedding auto probe requires CUDA.")
    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            0,
        )
    )
    device_count = torch.cuda.device_count()
    if local_rank >= device_count and device_count != 1:
        raise RuntimeError(
            "Vision patch embedding auto probe local rank exceeds visible CUDA devices."
        )
    device_index = local_rank if local_rank < device_count else 0
    torch.cuda.set_device(device_index)
    return torch.device(
        "cuda",
        device_index,
    )


def _verify_equivalence(
    signature: Dict[str, Any],
    device: torch.device,
    state_path: str,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> Dict[str, float]:
    torch.manual_seed(2026)
    convolution = _build_convolution(
        signature=signature,
        device=device,
        dtype=dtype,
    )
    input_shape = _build_input_shape(
        signature=signature,
        patch_count=2,
    )
    native_input = torch.randn(
        input_shape,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    linear_input = native_input.detach().clone().requires_grad_(True)
    linear_weight = convolution.weight.detach().clone().requires_grad_(True)
    linear_bias = (
        convolution.bias.detach().clone().requires_grad_(True)
        if convolution.bias is not None
        else None
    )
    _write_probe_phase(
        state_path=state_path,
        phase="linear_correctness",
    )
    linear_output = F.linear(
        linear_input.flatten(1),
        linear_weight.flatten(1),
        linear_bias,
    )
    linear_parameters: List[torch.Tensor] = [
        linear_input,
        linear_weight,
    ]
    if linear_bias is not None:
        linear_parameters.append(linear_bias)
    linear_gradients = torch.autograd.grad(
        outputs=linear_output.sum(),
        inputs=linear_parameters,
    )

    _write_probe_phase(
        state_path=state_path,
        phase="native_correctness",
    )
    native_output = convolution(native_input).flatten(1)
    native_parameters: List[torch.Tensor] = [
        native_input,
        convolution.weight,
    ]
    if convolution.bias is not None:
        native_parameters.append(convolution.bias)
    native_gradients = torch.autograd.grad(
        outputs=native_output.sum(),
        inputs=native_parameters,
    )

    _write_probe_phase(
        state_path=state_path,
        phase="equivalence_validation",
    )
    _require_close(
        name="output",
        observed=native_output,
        expected=linear_output,
        atol=atol,
        rtol=rtol,
    )
    gradient_differences = []
    for gradient_index, (native_gradient, linear_gradient) in enumerate(
        zip(
            native_gradients,
            linear_gradients,
            strict=True,
        )
    ):
        _require_close(
            name=f"gradient_{gradient_index}",
            observed=native_gradient,
            expected=linear_gradient,
            atol=atol,
            rtol=rtol,
        )
        gradient_differences.append(
            float((native_gradient - linear_gradient).abs().max().item())
        )

    return {
        "output_max_abs_difference": float(
            (native_output - linear_output).abs().max().item()
        ),
        "gradient_max_abs_difference": max(gradient_differences),
    }


def _measure_operators(
    signature: Dict[str, Any],
    probe_config: Dict[str, Any],
    device: torch.device,
    state_path: str,
) -> List[Dict[str, float]]:
    dtype = _resolve_torch_dtype(dtype_name=signature["dtype"])
    convolution = _build_convolution(
        signature=signature,
        device=device,
        dtype=dtype,
    )
    measurements = []
    with torch.inference_mode():
        for patch_count in probe_config["patch_counts"]:
            inputs = torch.randn(
                _build_input_shape(
                    signature=signature,
                    patch_count=int(patch_count),
                ),
                device=device,
                dtype=dtype,
            )
            _write_probe_phase(
                state_path=state_path,
                phase="native_benchmark",
            )
            native_times = _measure_callable(
                function=partial(
                    convolution,
                    inputs,
                ),
                device=device,
                warmup_iterations=int(probe_config["warmup_iterations"]),
                measurement_iterations=int(probe_config["measurement_iterations"]),
            )
            _write_probe_phase(
                state_path=state_path,
                phase="linear_benchmark",
            )
            linear_times = _measure_callable(
                function=partial(
                    _run_linear_projection,
                    inputs=inputs,
                    weight=convolution.weight,
                    bias=convolution.bias,
                ),
                device=device,
                warmup_iterations=int(probe_config["warmup_iterations"]),
                measurement_iterations=int(probe_config["measurement_iterations"]),
            )
            native_milliseconds = statistics.median(native_times) * 1000.0
            linear_milliseconds = statistics.median(linear_times) * 1000.0
            measurements.append(
                {
                    "patch_count": int(patch_count),
                    "native_milliseconds": native_milliseconds,
                    "linear_milliseconds": linear_milliseconds,
                    "slowdown_ratio": native_milliseconds / linear_milliseconds,
                }
            )
    return measurements


def _measure_callable(
    function: Callable[[], torch.Tensor],
    device: torch.device,
    warmup_iterations: int,
    measurement_iterations: int,
) -> List[float]:
    for _ in range(warmup_iterations):
        function()
    torch.cuda.synchronize(device=device)

    durations = []
    for _ in range(measurement_iterations):
        started_at = time.perf_counter()
        function()
        torch.cuda.synchronize(device=device)
        durations.append(time.perf_counter() - started_at)
    return durations


def _run_linear_projection(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    return F.linear(
        inputs.flatten(1),
        weight.flatten(1),
        bias,
    )


def _build_convolution(
    signature: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    convolution_class = nn.Conv2d if signature["dimension"] == 2 else nn.Conv3d
    convolution = convolution_class(
        in_channels=int(signature["in_channels"]),
        out_channels=int(signature["out_channels"]),
        kernel_size=tuple(signature["kernel_size"]),
        stride=tuple(signature["stride"]),
        padding=0,
        dilation=1,
        groups=1,
        bias=bool(signature["bias"]),
    )
    return convolution.to(
        device=device,
        dtype=dtype,
    )


def _build_input_shape(
    signature: Dict[str, Any],
    patch_count: int,
) -> Tuple[int, ...]:
    return (
        patch_count,
        int(signature["in_channels"]),
        *tuple(int(size) for size in signature["kernel_size"]),
    )


def _resolve_torch_dtype(
    dtype_name: str,
) -> torch.dtype:
    dtype_by_name = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported vision patch probe dtype: {dtype_name}.")
    return dtype_by_name[dtype_name]


def _require_close(
    name: str,
    observed: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> None:
    if torch.allclose(
        observed,
        expected,
        atol=atol,
        rtol=rtol,
    ):
        return
    maximum_difference = float((observed - expected).abs().max().item())
    raise RuntimeError(
        f"Vision patch embedding {name} equivalence failed: "
        f"max_abs_difference={maximum_difference}."
    )


def _read_json(
    path: str,
) -> Dict[str, Any]:
    with open(
        path,
        encoding="utf-8",
    ) as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Vision patch embedding probe result must be a mapping.")
    return payload


def _read_text(
    path: str,
) -> str:
    if not os.path.isfile(path):
        return ""
    with open(
        path,
        encoding="utf-8",
    ) as file:
        return file.read()


def _read_probe_phase(
    state_path: str,
) -> str:
    if not os.path.isfile(state_path):
        return "operation"
    payload = _read_json(path=state_path)
    phase = payload.get("phase")
    return (
        phase
        if isinstance(
            phase,
            str,
        )
        else "operation"
    )


def _write_probe_phase(
    state_path: str,
    phase: str,
) -> None:
    _write_json(
        path=state_path,
        payload={"phase": phase},
    )


def _write_json(
    path: str,
    payload: Dict[str, Any],
) -> None:
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    os.replace(
        temp_path,
        path,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Vision patch embedding probe requires one JSON payload.")
    _run_probe_child(payload=json.loads(sys.argv[1]))
