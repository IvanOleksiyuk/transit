from pathlib import Path
import shutil
import subprocess

import torch.fx as fx
from torch import nn


def visualize_fx_graph_png(
    model: nn.Module,
    out_path: str | Path,
    leaf_module_types: tuple[type[nn.Module], ...] = (),
) -> Path:
    """Trace a model with torch.fx and render the graph as PNG."""

    class _Tracer(fx.Tracer):
        def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
            if isinstance(m, leaf_module_types):
                return True
            return super().is_leaf_module(m, module_qualified_name)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_path.with_suffix(".dot")

    graph = _Tracer().trace(model)
    gm = fx.GraphModule(model, graph)

    lines = ["digraph FXGraph {", "  rankdir=LR;", "  node [shape=box];"]
    node_names: dict[fx.Node, str] = {}

    for idx, node in enumerate(gm.graph.nodes):
        node_id = f"n{idx}"
        node_names[node] = node_id
        if node.op == "call_module":
            label = f"{node.op}\\n{node.target}"
        elif node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            label = f"{node.op}\\n{name}"
        elif node.op == "call_method":
            label = f"{node.op}\\n{node.target}"
        else:
            label = f"{node.op}\\n{node.name}"
        lines.append(f'  {node_id} [label="{label}"];')

    for node in gm.graph.nodes:
        for in_node in node.all_input_nodes:
            lines.append(f"  {node_names[in_node]} -> {node_names[node]};")

    lines.append("}")
    dot_text = "\n".join(lines)
    dot_path.write_text(dot_text)

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        raise RuntimeError(
            f"Graphviz 'dot' executable not found. DOT graph saved to: {dot_path}"
        )

    subprocess.run(
        [dot_bin, "-Tpng", str(dot_path), "-o", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_path
