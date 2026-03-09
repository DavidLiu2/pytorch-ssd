import sys
import onnx

def build_consumers(graph):
    consumers = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return consumers

def strip_fake_quant(model):
    graph = model.graph
    consumers = build_consumers(graph)

    patterns = []  # (div_node, floor_node, mul_node, min_node, clip_node, src_name, dst_name)

    # First pass: discover patterns
    for node in graph.node:
        if node.op_type != "Div":
            continue

        div = node
        div_out = div.output[0]
        if div_out not in consumers or len(consumers[div_out]) != 1:
            continue
        floor = consumers[div_out][0]
        if floor.op_type != "Floor":
            continue

        floor_out = floor.output[0]
        if floor_out not in consumers or len(consumers[floor_out]) != 1:
            continue
        mul = consumers[floor_out][0]
        if mul.op_type != "Mul":
            continue

        mul_out = mul.output[0]
        if mul_out not in consumers or len(consumers[mul_out]) != 1:
            continue

        first = consumers[mul_out][0]
        if first.op_type == "Min":
            min_node = first
            min_out = min_node.output[0]
            if min_out not in consumers or len(consumers[min_out]) != 1:
                continue
            clip = consumers[min_out][0]
            if clip.op_type != "Clip":
                continue
        elif first.op_type == "Clip":
            min_node = None
            clip = first
        else:
            continue

        clip_out = clip.output[0]
        src = div.input[0]   # original (pre-quant) tensor
        dst = clip_out       # final (post-quant) tensor

        patterns.append((div, floor, mul, min_node, clip, src, dst))

    print(f"Found {len(patterns)} fake-quant patterns to strip")

    # Second pass: rewire consumers and mark nodes for removal
    nodes_to_remove_ids = set()

    for div, floor, mul, min_node, clip, src, dst in patterns:
        # Rewire all nodes that currently consume 'dst' to consume 'src' instead
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == dst:
                    node.input[i] = src

        nodes_to_remove_ids.add(id(div))
        nodes_to_remove_ids.add(id(floor))
        nodes_to_remove_ids.add(id(mul))
        if min_node is not None:
            nodes_to_remove_ids.add(id(min_node))
        nodes_to_remove_ids.add(id(clip))

    # Build new node list using ids
    old_count = len(graph.node)
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove_ids]
    removed_count = old_count - len(new_nodes)

    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"Removed {removed_count} nodes belonging to fake-quant chains")

    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python strip_fake_quant.py input.onnx output.onnx")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    model = strip_fake_quant(model)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print("Saved stripped model to", out)

if __name__ == "__main__":
    main()
