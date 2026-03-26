# Hybrid Follow Integer Add Audit

## Scale Selection

### post_qd
- stage4.0.add: eps_in=[1.1651866770989727e-05, 0.009759310632944107] eps_out=0.019518621265888214 D=65536 shift=16 mul=[39, 32768] requantization_factor=32
  branch_eps_ratio=837.5748560086854 output_lsb_per_input_lsb=[0.0005950927734375, 0.5] input_lsb_per_output_lsb=[1680.4102564102564, 2.0] forward_uses_requantization=0
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 478, 'end_line': 486} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161372669041157] eps_out=0.018322745338082314 D=65536 shift=16 mul=[49, 32768] requantization_factor=32
  branch_eps_ratio=663.4426032307229 output_lsb_per_input_lsb=[0.0007476806640625, 0.5] input_lsb_per_output_lsb=[1337.469387755102, 2.0] forward_uses_requantization=0
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 478, 'end_line': 486} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}

### post_id
- stage4.0.add: eps_in=[1.1651866770989727e-05, 0.009759310632944107] eps_out=0.019518621265888214 D=65536 shift=16 mul=[39, 32768] requantization_factor=32
  branch_eps_ratio=837.5748560086854 output_lsb_per_input_lsb=[0.0005950927734375, 0.5] input_lsb_per_output_lsb=[1680.4102564102564, 2.0] forward_uses_requantization=1
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 478, 'end_line': 486} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161372669041157] eps_out=0.018322745338082314 D=65536 shift=16 mul=[49, 32768] requantization_factor=32
  branch_eps_ratio=663.4426032307229 output_lsb_per_input_lsb=[0.0007476806640625, 0.5] input_lsb_per_output_lsb=[1337.469387755102, 2.0] forward_uses_requantization=1
  get_output_eps={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 478, 'end_line': 486} forward={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 488, 'end_line': 499} requant={'path': '/mnt/c/Users/yxl21/Documents/School/DroneRS/nemoenv/lib/python3.8/site-packages/nemo/quant/pact.py', 'start_line': 59, 'end_line': 70}
