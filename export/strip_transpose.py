import sys
import onnx

def strip_transpose(model):
    graph = model.graph

    patterns = []  # (transpose_node, src, dst)

    # Discover Transpose nodes
    for node in graph.node:
        if node.op_type != "Transpose":
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            # Unexpected pattern, skip for safety
            continue

        src = node.input[0]
        dst = node.output[0]
        patterns.append((node, src, dst))

    print(f"Found {len(patterns)} Transpose nodes to strip/bypass")

    nodes_to_remove_ids = set()

    # Rewire and mark for removal
    for tnode, src, dst in patterns:
        # Rewire all nodes that consume 'dst' to consume 'src' instead
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == dst:
                    node.input[i] = src

        nodes_to_remove_ids.add(id(tnode))

    old_count = len(graph.node)
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove_ids]
    removed_count = old_count - len(new_nodes)

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"Removed {removed_count} Transpose nodes")

    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python strip_transpose.py input.onnx output.onnx")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    model = strip_transpose(model)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print("Saved stripped model to", out)

if __name__ == "__main__":
    main()
