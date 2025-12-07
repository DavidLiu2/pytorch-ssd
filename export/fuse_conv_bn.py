import sys
import onnx
import numpy as np
from onnx import numpy_helper, TensorProto

def get_initializer_dict(graph):
    return {init.name: init for init in graph.initializer}

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.INT:
                return a.i
    return default

def fuse_conv_bn(model):
    graph = model.graph
    init_dict = get_initializer_dict(graph)

    new_nodes = []
    i = 0
    fused_count = 0

    while i < len(graph.node):
        node = graph.node[i]

        if (node.op_type == "Conv" and
            i + 1 < len(graph.node) and
            graph.node[i + 1].op_type == "BatchNormalization"):

            bn = graph.node[i + 1]

            # BN must directly consume Conv output
            if bn.input[0] != node.output[0]:
                new_nodes.append(node)
                i += 1
                continue

            # pull Conv weights/bias
            W_name = node.input[1]
            W_init = init_dict[W_name]
            W = numpy_helper.to_array(W_init)  # [out_c, in_c, kH, kW]

            if len(node.input) > 2:
                b_name = node.input[2]
                b_init = init_dict[b_name]
                b = numpy_helper.to_array(b_init)  # [out_c]
            else:
                b = np.zeros(W.shape[0], dtype=np.float32)
                # create a bias initializer
                b_name = node.name + "_bias_fused"
                b_init = numpy_helper.from_array(b, name=b_name)
                graph.initializer.append(b_init)
                node.input.append(b_name)
                init_dict[b_name] = b_init

            # pull BN params: scale, bias, mean, var
            scale = numpy_helper.to_array(init_dict[bn.input[1]])  # gamma
            bn_bias = numpy_helper.to_array(init_dict[bn.input[2]])  # beta
            mean = numpy_helper.to_array(init_dict[bn.input[3]])
            var = numpy_helper.to_array(init_dict[bn.input[4]])

            eps = get_attr(bn, "epsilon", 1e-5)

            # fuse: y_bn = gamma * (y - mean) / sqrt(var+eps) + beta
            denom = np.sqrt(var + eps)
            alpha = scale / denom           # per-channel multiplier
            beta = bn_bias - scale * mean / denom  # per-channel offset

            # W'[c] = W[c] * alpha[c]
            W_fused = W * alpha.reshape(-1, 1, 1, 1)
            # b'[c] = alpha[c] * b[c] + beta[c]
            b_fused = alpha * b + beta

            # write back
            W_init.CopyFrom(numpy_helper.from_array(W_fused.astype(np.float32), name=W_name))
            b_init.CopyFrom(numpy_helper.from_array(b_fused.astype(np.float32), name=b_name))

            # Conv should now output what BN used to output
            conv_out = node.output[0]
            bn_out = bn.output[0]
            if bn_out != conv_out:
                # redirect Conv's output name to BN's output name
                node.output[0] = bn_out

            new_nodes.append(node)
            fused_count += 1

            # skip BN
            i += 2
        else:
            new_nodes.append(node)
            i += 1

    # replace node list
    del graph.node[:]
    graph.node.extend(new_nodes)

    print(f"Fused {fused_count} Conv+BN pairs")
    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python fuse_conv_bn.py input.onnx output.onnx")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    model = fuse_conv_bn(model)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print("Saved fused model to", out)

if __name__ == "__main__":
    main()
