# Hybrid Follow Integer Add Audit

- Integer add scale policy: `fanin`

## Scale Selection

### post_qd
- stage4.0.add: eps_in=[1.165186313301092e-05, 3.091901453444734e-05] eps_out=6.183802906889468e-05 D=256 shift=8 mul=[48, 128] requantization_factor=32
  branch_eps_ratio=2.653568290452246 output_lsb_per_input_lsb=[0.1875, 0.5] input_lsb_per_output_lsb=[5.333333333333333, 2.0] forward_uses_requantization=0
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161373600363731] eps_out=0.018322747200727463 D=65536 shift=16 mul=[49, 32768] requantization_factor=32
  branch_eps_ratio=663.4426706746633 output_lsb_per_input_lsb=[0.0007476806640625, 0.5] input_lsb_per_output_lsb=[1337.469387755102, 2.0] forward_uses_requantization=0
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}

### post_id
- stage4.0.add: eps_in=[1.165186313301092e-05, 3.091901453444734e-05] eps_out=6.183802906889468e-05 D=256 shift=8 mul=[48, 128] requantization_factor=32
  branch_eps_ratio=2.653568290452246 output_lsb_per_input_lsb=[0.1875, 0.5] input_lsb_per_output_lsb=[5.333333333333333, 2.0] forward_uses_requantization=1
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}
- stage4.1.add: eps_in=[1.3808839867124334e-05, 0.009161373600363731] eps_out=0.018322747200727463 D=65536 shift=16 mul=[49, 32768] requantization_factor=32
  branch_eps_ratio=663.4426706746633 output_lsb_per_input_lsb=[0.0007476806640625, 0.5] input_lsb_per_output_lsb=[1337.469387755102, 2.0] forward_uses_requantization=1
  get_output_eps={'path': 'C:\\Users\\yxl21\\Documents\\School\\DroneRS\\pytorch_ssd\\export_nemo_quant.py', 'start_line': 180, 'end_line': 193} forward={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 485, 'end_line': 496} requant={'path': 'C:\\Users\\yxl21\\AppData\\Roaming\\Python\\Python313\\site-packages\\nemo\\quant\\pact.py', 'start_line': 57, 'end_line': 67}
