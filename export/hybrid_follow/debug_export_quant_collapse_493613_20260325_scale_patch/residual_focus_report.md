# Hybrid Follow Residual Drift Focus

- Largest FQ->ID drift point: `stage4.1.add pre-requant`
- Largest FQ->ID drift mean abs diff: `1.296827`
- stage4.1.add pre->post requant mean abs diff: `0.006574`
- stage4.1.add eps_in=[1.3808839867124334e-05, 0.009161372669041157] eps_out=0.012956136837601662 D=32768 shift=15 mul=[35, 23170]

Largest inspected FQ->ID drift is at 'stage4.1.add pre-requant' with mean_abs_diff=1.296827. The stage4.1.add requant step alone changes the semantic tensor by mean_abs_diff=0.006574. Branch output LSB per input LSB is [0.001068115234375, 0.70709228515625] with input LSB per output LSB [936.2285714285714, 1.4142425550280535].
