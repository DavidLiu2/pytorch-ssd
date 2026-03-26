# Hybrid Follow Integer Add Audit

- Integer add scale policy: `sqrt_fanin`

## Scale Selection

### post_qd
- stage4.0.add: eps_in=[1.1651866770989727e-05, 0.009759310632944107] eps_out=0.013801748864352703 D=65536 shift=16 mul=[55, 46341] requantization_factor=32
  branch_eps_ratio=837.5748560086854 output_lsb_per_input_lsb=[0.0008392333984375, 0.7071075439453125] input_lsb_per_output_lsb=[1191.5636363636363, 1.4142120368572106] forward_uses_requantization=0
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export_nemo_quant.py', 'start_line': 173, 'end_line': 186} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161372669041157] eps_out=0.012956136837601662 D=32768 shift=15 mul=[35, 23170] requantization_factor=32
  branch_eps_ratio=663.4426032307229 output_lsb_per_input_lsb=[0.001068115234375, 0.70709228515625] input_lsb_per_output_lsb=[936.2285714285714, 1.4142425550280535] forward_uses_requantization=0
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export_nemo_quant.py', 'start_line': 173, 'end_line': 186} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}

### post_id
- stage4.0.add: eps_in=[1.1651866770989727e-05, 0.009759310632944107] eps_out=0.013801748864352703 D=65536 shift=16 mul=[55, 46341] requantization_factor=32
  branch_eps_ratio=837.5748560086854 output_lsb_per_input_lsb=[0.0008392333984375, 0.7071075439453125] input_lsb_per_output_lsb=[1191.5636363636363, 1.4142120368572106] forward_uses_requantization=1
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export_nemo_quant.py', 'start_line': 173, 'end_line': 186} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161372669041157] eps_out=0.012956136837601662 D=32768 shift=15 mul=[35, 23170] requantization_factor=32
  branch_eps_ratio=663.4426032307229 output_lsb_per_input_lsb=[0.001068115234375, 0.70709228515625] input_lsb_per_output_lsb=[936.2285714285714, 1.4142425550280535] forward_uses_requantization=1
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/pytorch_ssd/export_nemo_quant.py', 'start_line': 173, 'end_line': 186} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}
