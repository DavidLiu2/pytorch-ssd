# Hybrid Follow Residual Drift Focus

- Largest FQ->ID drift point: `stage4.1.add pre-requant`
- Largest FQ->ID drift mean abs diff: `0.391293`
- stage4.1.add pre->post requant mean abs diff: `0.009202`
- stage4.1.add eps_in=[1.3808839867124334e-05, 0.009161373600363731] eps_out=0.018322747200727463 D=65536 shift=16 mul=[49, 32768]

Largest inspected FQ->ID drift is at 'stage4.1.add pre-requant' with mean_abs_diff=0.391293. The stage4.1.add requant step alone changes the semantic tensor by mean_abs_diff=0.009202. Branch output LSB per input LSB is [0.0007476806640625, 0.5] with input LSB per output LSB [1337.469387755102, 2.0].
