import sys
import onnx

def get_initializer_names(graph):
    return {init.name for init in graph.initializer}

def strip_min(model):
    graph = model.graph
    init_names = get_initializer_names(graph)

    patterns = []  # (min_node, src, dst)

    # Discover Min nodes
    for node in graph.node:
        if node.op_type != "Min":
            continue

        if len(node.input) != 2:
            # Unexpected pattern, skip for safety
            continue

        in0, in1 = node.input
        out = node.output[0]

        # Heuristic: one input is a Constant (initializer), the other is the "real" tensor
        if in0 in init_names and in1 not in init_names:
            src = in1
        elif in1 in init_names and in0 not in init_names:
            src = in0
        else:
            # Both or neither are constants -> skip this Min for now
            continue

        dst = out
        patterns.append((node, src, dst))

    print(f"Found {len(patterns)} Min nodes to strip/bypass")

    # Rewire and mark for removal
    nodes_to_remove_ids = set()

    for min_node, src, dst in patterns:
        # Rewire all nodes that consume 'dst' to consume 'src' instead
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == dst:
                    node.input[i] = src

        nodes_to_remove_ids.add(id(min_node))

    old_count = len(graph.node)
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove_ids]
    removed_count = old_count - len(new_nodes)

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"Removed {removed_count} Min nodes")

    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python strip_min.py input.onnx output.onnx")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    model = strip_min(model)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print("Saved stripped model to", out)

if __name__ == "__main__":
    main()
