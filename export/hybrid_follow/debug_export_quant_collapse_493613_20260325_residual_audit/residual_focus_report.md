# Hybrid Follow Residual Drift Focus

- Largest FQ->ID drift point: `stage4.1.add pre-requant`
- Largest FQ->ID drift mean abs diff: `1.301144`
- stage4.1.add pre->post requant mean abs diff: `0.008937`
- stage4.1.add eps_in=[1.3808839867124334e-05, 0.009161372669041157] eps_out=0.018322745338082314 D=65536 shift=16 mul=[49, 32768]

Largest inspected FQ->ID drift is at 'stage4.1.add pre-requant' with mean_abs_diff=1.301144. The stage4.1.add requant step alone changes the semantic tensor by mean_abs_diff=0.008937. Branch output LSB per input LSB is [0.0007476806640625, 0.5] with input LSB per output LSB [1337.469387755102, 2.0].
