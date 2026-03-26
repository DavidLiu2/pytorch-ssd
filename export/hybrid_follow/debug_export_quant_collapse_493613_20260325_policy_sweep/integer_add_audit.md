# Hybrid Follow Integer Add Audit

- Integer add scale policy: `sqrt_fanin`

## Scale Selection

### post_qd
- stage4.0.add: eps_in=[1.165186313301092e-05, 3.091901453444734e-05] eps_out=4.372608964331448e-05 D=128 shift=7 mul=[34, 91] requantization_factor=32
  branch_eps_ratio=2.653568290452246 output_lsb_per_input_lsb=[0.265625, 0.7109375] input_lsb_per_output_lsb=[3.764705882352941, 1.4065934065934067] forward_uses_requantization=0
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161373600363731] eps_out=0.012956138700246811 D=32768 shift=15 mul=[35, 23170] requantization_factor=32
  branch_eps_ratio=663.4426706746633 output_lsb_per_input_lsb=[0.001068115234375, 0.70709228515625] input_lsb_per_output_lsb=[936.2285714285714, 1.4142425550280535] forward_uses_requantization=0
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}

### post_id
- stage4.0.add: eps_in=[1.165186313301092e-05, 3.091901453444734e-05] eps_out=4.372608964331448e-05 D=128 shift=7 mul=[34, 91] requantization_factor=32
  branch_eps_ratio=2.653568290452246 output_lsb_per_input_lsb=[0.265625, 0.7109375] input_lsb_per_output_lsb=[3.764705882352941, 1.4065934065934067] forward_uses_requantization=1
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161373600363731] eps_out=0.012956138700246811 D=32768 shift=15 mul=[35, 23170] requantization_factor=32
  branch_eps_ratio=663.4426706746633 output_lsb_per_input_lsb=[0.001068115234375, 0.70709228515625] input_lsb_per_output_lsb=[936.2285714285714, 1.4142425550280535] forward_uses_requantization=1
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}
