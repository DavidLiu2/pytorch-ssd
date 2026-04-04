import sys

import onnx


def simplify_onnx(model):
    try:
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError(
            "onnxsim is required for simplify_onnx.py. Run this step from the nemoenv toolchain."
        ) from exc

    simplified, ok = simplify(model)
    if not ok:
        raise RuntimeError("onnxsim returned ok=False while simplifying the ONNX graph.")
    return simplified


def main():
    if len(sys.argv) != 3:
        print("Usage: python simplify_onnx.py input.onnx output.onnx")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    model = onnx.load(inp)
    model = simplify_onnx(model)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print("Saved simplified model to", out)


if __name__ == "__main__":
    main()
