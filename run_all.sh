#!/bin/bash
exec > log.txt 2>&1
set -e

########################################
# PROJECT & ENV PATHS
########################################

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DORY_ENV_DIR="${PROJECT_ROOT}/../doryenv"
NEMO_ENV_DIR="${PROJECT_ROOT}/../nemoenv"

DORY_REQ="${PROJECT_ROOT}/requirements_doryenv.txt"
NEMO_REQ="${PROJECT_ROOT}/requirements_nemoenv.txt"

# ONNX filenames
FUSED_ONNX="export/ssd_mbv2_fused.onnx"
SIMPLIFIED_ONNX="export/ssd_mbv2_simplified.onnx"
NO_TRANSPOSE_ONNX="export/ssd_mbv2_notranspose.onnx"
NO_MIN_ONNX="export/ssd_mbv2_nomin.onnx"
FINAL_DORY_ONNX="export/ssd_mbv2_dory.onnx"

########################################
# HELPERS
########################################

ensure_venv() {
  local env_dir="$1"
  local req_file="$2"

  if [ ! -d "$env_dir" ]; then
    echo "=== Creating virtualenv at ${env_dir} ==="
    python3 -m venv "$env_dir"
    # shellcheck disable=SC1090
    source "$env_dir/bin/activate"
    pip install --upgrade pip
    if [ -f "$req_file" ]; then
      pip install -r "$req_file"
    else
      echo "WARNING: requirements file not found: $req_file"
    fi
    deactivate
  fi
}

########################################
# ENSURE ENVS EXIST
########################################

ensure_venv "$DORY_ENV_DIR" "$DORY_REQ"
ensure_venv "$NEMO_ENV_DIR" "$NEMO_REQ"

########################################
# 1. TRAIN (whatever env you started in, usually doryenv)
########################################

echo "=== [1/7] Training SSD-MobileNetV2 ==="
# python3 train.py

########################################
# 2. EXPORT RAW WRAPPER CHECKPOINT FOR NEMO
########################################

echo "=== [2/7] Building raw wrapper checkpoint for NEMO ==="
python3 export_float_raw.py
# this script should save training/person_ssd_pytorch/ssd_mbv2_raw.pth

########################################
# 3. NEMO QUANTIZATION → QUANT ONNX (IN nemoenv)
########################################

echo "=== [3/7] Exporting NEMO-quantized ONNX (using nemoenv) ==="

ORIG_VENV="$VIRTUAL_ENV"   # remember where we started

# activate nemoenv if not already there
if [ "$VIRTUAL_ENV" != "$NEMO_ENV_DIR" ]; then
  # shellcheck disable=SC1090
  source "$NEMO_ENV_DIR/bin/activate"
  echo "activated nemoenv: $NEMO_ENV_DIR"
fi

# run NEMO export (this should write export/ssd_mbv2_quant.onnx)
CUDA_VISIBLE_DEVICES="" python3 export_nemo_quant.py

# quantized ONNX path (export_nemo_quant.py's default output)
QUANT_ONNX="export/ssd_mbv2_quant.onnx"

# leave nemoenv
if [ "$VIRTUAL_ENV" = "$NEMO_ENV_DIR" ]; then
  deactivate
fi

# restore previous env (typically ../doryenv for you)
if [ -n "$ORIG_VENV" ] && [ -f "$ORIG_VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$ORIG_VENV/bin/activate"
  echo "activated original venv: $ORIG_VENV"
elif [ -d "$DORY_ENV_DIR" ]; then
  # fallback: go to doryenv
  # shellcheck disable=SC1090
  source "$DORY_ENV_DIR/bin/activate"
  echo "activated original venv: $DORY_ENV_DIR"
fi

########################################
# 4. ONNX SIMPLIFIER (onnx-sim) BEFORE STRIPPING
########################################

echo "=== [4/7] Simplifying ONNX with onnx-simplifier ==="
python3 -m onnxsim \
  "${QUANT_ONNX}" \
  "${SIMPLIFIED_ONNX}" \
  --skip-optimization


########################################
# 5. FUSE CONV + BN (ON QUANT ONNX)
########################################

echo "=== [5/7] Fusing Conv + BatchNorm in ONNX graph ==="
python3 export/fuse_conv_bn.py \
  "${SIMPLIFIED_ONNX}" \
  "${FUSED_ONNX}"


########################################
# 6. STRIP UNSUPPORTED / UGLY OPS (AFTER onnx-sim)
########################################

echo "=== [6a] Stripping Transpose nodes ==="
python3 export/strip_transpose.py \
  "${FUSED_ONNX}" \
  "${NO_TRANSPOSE_ONNX}"

echo "=== [6b] Stripping Min nodes ==="
python3 export/strip_min.py \
  "${NO_TRANSPOSE_ONNX}" \
  "${NO_MIN_ONNX}"

echo "=== [6c] Stripping FakeQuant nodes ==="
python3 export/strip_fake_quant.py \
  "${NO_MIN_ONNX}" \
  "${FINAL_DORY_ONNX}"

echo "DORY-ready ONNX: ${FINAL_DORY_ONNX}"



########################################
# 7. DORY CODE GENERATION (typically in doryenv)
########################################

echo "=== [7/7] Running DORY network_generate ==="
# if you keep DORY repo somewhere else, adjust this path + config
(
  cd ../dory || exit 1
  python3 network_generate.py \
    NEMO \
    PULP.GAP8 \
    ../dory_examples/config_files/config_person_ssd.json \
    --n_inputs 0
)

# may need to patch dory/dory/Parsers/HW_node.py add_checksum_activations_integer method as follows:
#     def add_checksum_activations_integer(self, load_directory, node_number, n_inputs=1):
        # ###########################################################################
        # ###### SECTION 4: GENERATE CHECKSUM BY USING OUT_LAYER{i}.TXT FILES  ######
        # ###########################################################################
        # self.check_sum_in = []
        # self.check_sum_out = []

        # for in_idx in range(n_inputs):
        #     # ---------- INPUT ACTIVATIONS ----------
        #     if node_number == 0:
        #         infile = 'input.txt' if n_inputs == 1 else f'input_{in_idx}.txt'
        #         in_path = os.path.join(load_directory, infile)
        #         try:
        #             try:
        #                 x = np.loadtxt(in_path, delimiter=',', dtype=np.uint8, usecols=[0])
        #             except ValueError:
        #                 x = np.loadtxt(in_path, delimiter=',', dtype=np.float, usecols=[0]).astype(np.int64)
        #             x = x.ravel()
        #             if self.input_activation_bits <= 8:
        #                 x = self._compress(x, self.input_activation_bits)
        #         except FileNotFoundError:
        #             # No input.txt -> generate random input (original behavior)
        #             print("========= WARNING ==========")
        #             print(f"Input file {in_path} not found; generating random inputs!")
        #             x = np.random.randint(
        #                 low=0,
        #                 high=2**8 - 1,
        #                 size=self.input_channels * self.input_dimensions[0] * self.input_dimensions[1],
        #                 dtype=np.uint8,
        #             )
        #             if self.input_activation_bits <= 8:
        #                 x = self._compress(x, self.input_activation_bits)
        #     else:
        #         infile = f'out_layer{node_number-1}.txt' if n_inputs == 1 else f'out_{in_idx}_layer{node_number-1}.txt'
        #         in_path = os.path.join(load_directory, infile)
        #         try:
        #             x = np.loadtxt(in_path, delimiter=',', dtype=np.int64, usecols=[0])
        #         except FileNotFoundError:
        #             # Missing previous layer output: just fake zeros for checksum
        #             print("========= WARNING ==========")
        #             print(f"Activation file {in_path} not found; using zeros as input checksum for node {node_number}.")
        #             n_elems_in = self.input_channels * self.input_dimensions[0] * self.input_dimensions[1]
        #             x = np.zeros(n_elems_in, dtype=np.int64)
        #         except ValueError:
        #             x = np.loadtxt(in_path, delimiter=',', dtype=np.float, usecols=[0]).astype(np.int64)

        #         if self.input_activation_bits <= 8:
        #             x = self._compress(x.ravel(), self.input_activation_bits)

        #     self.check_sum_in.append(int(np.sum(x)))

        #     # ---------- OUTPUT ACTIVATIONS ----------
        #     outfile = f'out_layer{node_number}.txt' if n_inputs == 1 else f'out_{in_idx}_layer{node_number}.txt'
        #     out_path = os.path.join(load_directory, outfile)

        #     try:
        #         y = np.loadtxt(out_path, delimiter=',', dtype=np.int64, usecols=[0])
        #     except FileNotFoundError:
        #         # This is the case that was crashing you before
        #         print("========= WARNING ==========")
        #         print(f"Activation file {out_path} not found; using zeros as output checksum for node {node_number}.")
        #         n_elems_out = self.output_channels * self.output_dimensions[0] * self.output_dimensions[1]
        #         y = np.zeros(n_elems_out, dtype=np.int64)
        #     except ValueError:
        #         y = np.loadtxt(out_path, delimiter=',', dtype=np.float, usecols=[0]).astype(np.int64)

        #     if self.output_activation_bits <= 8:
        #         y = self._compress(y.ravel(), self.output_activation_bits)
        #     elif self.split_ints and self.output_activation_bits > 8:
        #         y = self._to_uint8(y.ravel(), self.output_activation_bits)

        #     self.check_sum_out.append(int(np.sum(y)))


# also replace with dory/dory/Utils/Templates_writer/Layer2D_template_writer.py", line 227 with
    # if "Addition" in node.name:
    #     # Second input quant info (should exist if this is a real Add node)
    #     ds_x2 = getattr(node, "second_input_activation_bits", None)
    #     dt_x2 = getattr(node, "second_input_activation_type", None)
    #     tk["data_type_x2"] = dt_x2
    #     tk["x_data_size_byte2"] = ds_x2

    #     # Helper to safely extract "value" from dict-like attrs
    #     def _get_qattr(n, attr_name, default):
    #         attr = getattr(n, attr_name, None)
    #         if isinstance(attr, dict) and "value" in attr:
    #             return attr["value"]
    #         print(f"[DORY] WARNING: node {getattr(n, 'name', 'unnamed')} has no {attr_name}; using {default} as default.")
    #         return default

    #     # Neutral defaults: mul=1, add=0, shift=0
    #     tk["inmul1"]   = _get_qattr(node, "inmul1",   1)
    #     tk["inadd1"]   = _get_qattr(node, "inadd1",   0)
    #     tk["inshift1"] = _get_qattr(node, "inshift1", 0)

    #     tk["inmul2"]   = _get_qattr(node, "inmul2",   1)
    #     tk["inadd2"]   = _get_qattr(node, "inadd2",   0)
    #     tk["inshift2"] = _get_qattr(node, "inshift2", 0)

    #     tk["outmul"]   = _get_qattr(node, "outmul",   1)
    #     tk["outadd"]   = _get_qattr(node, "outadd",   0)
    #     tk["outshift"] = _get_qattr(node, "outshift", 0)


# patch dory/dory/Hardware_targets/PULP/GAP8_L2/C_Parser.py", create_hex_input method as follows:
    # def create_hex_input(self):
        # print("\nGenerating .h input file.")
        # prefix = self.HWgraph[0].prefix
        # x_in_l = []

        # # Ensure group exists (DORY uses it in other places)
        # if not hasattr(self, "group"):
        #     self.group = 1

        # for in_idx in range(self.n_inputs):
        #     infile = 'input.txt' if self.n_inputs == 1 else f'input_{in_idx}.txt'
        #     try:
        #         # Try to load user-provided input
        #         x_in = np.loadtxt(
        #             os.path.join(self.network_directory, infile),
        #             delimiter=',',
        #             dtype=np.uint8,
        #             usecols=[0],
        #         )
        #     except FileNotFoundError:
        #         # If missing, just generate a tiny dummy input
        #         print(
        #             "========= WARNING ==========\n"
        #             f"Input file {os.path.join(self.network_directory, 'input.txt')} not found; "
        #             "generating dummy random input of length 1!"
        #         )
        #         x_in = np.random.randint(
        #             low=0,
        #             high=2**8,   # 0–255
        #             size=1,      # <<< no dependence on channels/height/width
        #             dtype=np.uint8,
        #         )

        #     x_in_l.append(x_in.flatten())

        # # Concatenate inputs (if n_inputs > 1)
        # x_in = np.concatenate(x_in_l)

        # # Try to get input bitwidth; fall back to 8 if not present
        # in_node = self.HWgraph[0]
        # in_bits = getattr(in_node, "input_activation_bits", 8)
        # if in_bits != 8:
        #     x_in = HW_node._compress(x_in, in_bits)

        # temp = x_in
        # input_values = utils.print_test_vector(temp.flatten(), 'char')
        # tk = OrderedDict([])
        # tk['input_values'] = input_values
        # tk['dimension'] = len(x_in)
        # tk['sdk'] = self.HW_description["software development kit"]["name"]
        # tk['prefix'] = prefix
        # root = os.path.dirname(__file__)
        # tmpl = Template(filename=os.path.join(root, "Templates/input_h_template.h"))
        # s = tmpl.render(**tk)
        # save_string = os.path.join(self.inc_dir, prefix + 'input.h')
        # with open(save_string, "w") as f:
        #     f.write(s)



echo "============================================="
echo " DONE: SSD-MBV2 quantized & cleaned for GAP8 "
echo "============================================="
