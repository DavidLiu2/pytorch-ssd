import sys
from collections import defaultdict

import onnx


PASSTHROUGH_OPS = {"Cast", "Identity"}


def _build_initializer_set(graph):
    return {init.name for init in graph.initializer}


def _build_consumers(graph):
    consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    return consumers


def _downstream_user_ops(tensor_name, consumers, *, seen_tensors=None, seen_nodes=None):
    if seen_tensors is None:
        seen_tensors = set()
    if seen_nodes is None:
        seen_nodes = set()
    if tensor_name in seen_tensors:
        return set()
    seen_tensors.add(tensor_name)

    ops = set()
    for node in consumers.get(tensor_name, []):
        if id(node) in seen_nodes:
            continue
        seen_nodes.add(id(node))
        ops.add(node.op_type)
        if node.op_type in PASSTHROUGH_OPS and len(node.output) == 1:
            ops |= _downstream_user_ops(
                node.output[0],
                consumers,
                seen_tensors=seen_tensors,
                seen_nodes=seen_nodes,
            )
    return ops


def _prune_unused_initializers(graph):
    used = set()
    for node in graph.node:
        for inp in node.input:
            if inp:
                used.add(inp)
    for out in graph.output:
        if out.name:
            used.add(out.name)

    kept = [init for init in graph.initializer if init.name in used]
    removed = len(graph.initializer) - len(kept)
    if removed:
        del graph.initializer[:]
        graph.initializer.extend(kept)
    return removed


def _pick_non_const_input(inputs, init_names):
    if len(inputs) != 2:
        return None
    a, b = inputs
    if a in init_names and b not in init_names:
        return b
    if b in init_names and a not in init_names:
        return a
    return None


def strip_affine_mul_add(model):
    graph = model.graph
    init_names = _build_initializer_set(graph)
    consumers = _build_consumers(graph)

    # Remove only affine pairs that are not part of BNRelu/Requant chains.
    # Candidate:
    #   Mul(data, const) -> Add(mul_out, const) -> {not Mul/Div/Clip/Floor}
    patterns = []  # (mul_node, add_node, src_tensor, add_out, user_ops)

    for mul_node in graph.node:
        if mul_node.op_type != "Mul" or len(mul_node.output) != 1:
            continue

        src_tensor = _pick_non_const_input(list(mul_node.input), init_names)
        if src_tensor is None:
            continue

        mul_out = mul_node.output[0]
        mul_users = consumers.get(mul_out, [])
        if len(mul_users) != 1:
            continue

        add_node = mul_users[0]
        if add_node.op_type != "Add" or len(add_node.input) != 2 or len(add_node.output) != 1:
            continue

        # Ensure the Add is a tensor + constant affine op.
        if add_node.input[0] == mul_out:
            other_in = add_node.input[1]
        elif add_node.input[1] == mul_out:
            other_in = add_node.input[0]
        else:
            continue
        if other_in not in init_names:
            continue

        add_out = add_node.output[0]
        user_ops = _downstream_user_ops(add_out, consumers)

        # Keep BNRelu/Requant patterns intact, even if a Cast/Identity sits between
        # the affine Add and the requantization chain.
        if any(op in {"Mul", "Div", "Clip", "Floor"} for op in user_ops):
            continue

        patterns.append((mul_node, add_node, src_tensor, add_out, user_ops))

    print(f"Found {len(patterns)} affine Mul->Add pairs to bypass")

    nodes_to_remove_ids = set()
    for mul_node, add_node, src_tensor, add_out, _ in patterns:
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == add_out:
                    node.input[i] = src_tensor
        for out in graph.output:
            if out.name == add_out:
                out.name = src_tensor

        nodes_to_remove_ids.add(id(mul_node))
        nodes_to_remove_ids.add(id(add_node))

    old_count = len(graph.node)
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove_ids]
    removed_count = old_count - len(new_nodes)

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"Removed {removed_count} nodes from affine Mul->Add pairs")
    pruned_initializers = _prune_unused_initializers(graph)
    if pruned_initializers:
        print(f"Pruned {pruned_initializers} unused initializers")
    return model


def main():
    if len(sys.argv) != 3:
        print("Usage: python strip_affine_mul_add.py input.onnx output.onnx")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    model = strip_affine_mul_add(model)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print("Saved stripped model to", out)


if __name__ == "__main__":
    main()
