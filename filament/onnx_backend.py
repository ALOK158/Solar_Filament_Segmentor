"""ONNX export of a member and ONNX Runtime inference.

The ONNX path replaces ONLY the network forward: `OrtSegmenter` is called exactly like the PyTorch module inside
inference.predict_maps, so preprocessing, TTA, averaging, decoding and filtering are shared and any difference
between the backends is the forward pass alone.

Measured on a Kaggle T4 over the 180 test images, stage 1 only, before the refiner was added (3 members x 4 views,
artifacts/onnx_benchmark.json):
PyTorch fp32 4.91 s/image; ONNX Runtime fp32 5.20 s/image (0.94x) with 100% identical submission rows;
mixed fp16 decoder convolutions 1.04x in PyTorch, with 92.9% identical rows. The network was 79% of that time, so
no backend change could buy more than ~1.3x: PyTorch fp32 ships, and this module documents the portable export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, path: str | Path, input_size: int = 1536, opset: int = 17) -> Path:
    """Static-shape export: images (1, 3, S, S) float32 -> logits (1, 2, S, S). A static shape lets ONNX Runtime
    plan the whole graph once; the pipeline always feeds the same size."""
    model = model.eval()
    sample = torch.zeros(1, 3, input_size, input_size, device=next(model.parameters()).device)
    kwargs = dict(input_names=["images"], output_names=["logits"], opset_version=opset, do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, (sample,), str(path), dynamo=False, **kwargs)
        except TypeError:                                   # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(model, (sample,), str(path), **kwargs)
    return Path(path)


def check_onnx(path: str | Path) -> dict:
    """Structural check of the exported graph."""
    import onnx

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return {"opset": graph.opset_import[0].version, "nodes": len(graph.graph.node),
            "inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.graph.input],
            "megabytes": round(Path(path).stat().st_size / 2 ** 20, 1)}


class OrtSegmenter:
    """Callable like the PyTorch member: (1, 3, S, S) float32 tensor on the pipeline's device -> logits there.
    On CUDA, input and output stay on the GPU (IO binding): no host round trip."""

    def __init__(self, path: str | Path, device: torch.device | str = "cpu") -> None:
        import onnxruntime as ort

        self.device = torch.device(device)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.device.type == "cuda":
            cuda = {"device_id": self.device.index or 0, "arena_extend_strategy": "kSameAsRequested"}
            providers = [("CUDAExecutionProvider", cuda), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        output = self.session.get_outputs()[0]
        self.output_name, self.output_shape = output.name, tuple(output.shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().float()
        if self.provider != "CUDAExecutionProvider":
            return torch.from_numpy(self.session.run([self.output_name], {self.input_name: x.cpu().numpy()})[0]).to(x.device)
        out = torch.empty(self.output_shape, dtype=torch.float32, device=x.device)
        binding = self.session.io_binding()
        binding.bind_input(self.input_name, "cuda", x.device.index or 0, np.float32, tuple(x.shape), x.data_ptr())
        binding.bind_output(self.output_name, "cuda", out.device.index or 0, np.float32, tuple(out.shape),
                            out.data_ptr())
        torch.cuda.current_stream().synchronize()          # x was written on torch's stream; ORT uses its own
        self.session.run_with_iobinding(binding)
        return out
