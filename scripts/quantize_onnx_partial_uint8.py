import onnx
import argparse
from collections import Counter
from onnxruntime.quantization import (
    preprocess,
    quantize_dynamic,
    QuantType
)

ORT_QUANTIZABLE_OPS = {
    "Conv",
    "MatMul",
    "Gemm",
    "Attention",
    "LSTM",
    "GRU",
    "Add",
    "Mul",
    "Gather"
}

def quantize_model(
    model_path: str,
    output_path: str,
    op_types_to_quantize=None,
    nodes_to_exclude=None
):
    """
    Quantize an ONNX model partially to INT8.

    Args:
        model_path: Path to FP32 ONNX model
        output_path: Path to save INT8 model
        op_types_to_quantize: List of op types to quantize (e.g. ["MatMul", "Conv"])
        nodes_to_exclude: List of node names to exclude from quantization
    """
    print(f"Quantizing model: {model_path}")
    print(f"Saving quantized model to: {output_path}")

    # Preprocessing model
    preprocessed_model_path = model_path.replace(".onnx", "_float32.onnx")
    preprocess.quant_pre_process(
        input_model_path=model_path,
        output_model_path=preprocessed_model_path,
        skip_symbolic_shape=True
    )

    quantize_dynamic(
        model_input=preprocessed_model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=op_types_to_quantize,
        nodes_to_exclude=nodes_to_exclude
    )

    print("Quantization complete.")


def list_ops(model_path):
    model = onnx.load(model_path)
    ops = [node.op_type for node in model.graph.node]
    counts = Counter(ops)

    print(f"\nOperations in model: {model_path}\n")
    print(f"{'OpType':25} {'Count':>6} {'Quantizable'}")
    print("-" * 45)

    for op, count in sorted(counts.items()):
        quantizable = "YES" if op in ORT_QUANTIZABLE_OPS else "NO"
        print(f"{op:25} {count:6} {quantizable}")

    print("\nSummary:")
    print(f"Total ops: {len(ops)}")
    print(f"Quantizable ops: {sum(counts[o] for o in ORT_QUANTIZABLE_OPS if o in counts)}")


def main():
    parser = argparse.ArgumentParser(description="Partial INT8 quantization for ONNX models")
    parser.add_argument(
        "--output",
        help="Path to output INT8 ONNX model (default: <model>_int8.onnx)"
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=ORT_QUANTIZABLE_OPS,
        help="ONNX op types to quantize"
    )
    parser.add_argument(
        "--exclude-nodes",
        nargs="+",
        default=[],
        help="Node names to exclude from quantization"
    )

    args = parser.parse_args()
    args.model = input("Path to model: ")

    model_path = args.model
    output_path = args.output or model_path.replace(".onnx", "_uint8.onnx")

    list_ops(args.model)

    quantize_model(
        model_path=model_path,
        output_path=output_path,
        op_types_to_quantize=args.ops,
        nodes_to_exclude=args.exclude_nodes
    )


if __name__ == "__main__":
    main()
