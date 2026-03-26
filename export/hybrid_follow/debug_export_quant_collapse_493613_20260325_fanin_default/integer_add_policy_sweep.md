# Hybrid Follow Integer Add Policy Sweep

- Active policy: `fanin`
- Selected policy: `fanin`
- Selection metric: `score_final_output = x_abs_diff + size_abs_diff + vis_conf_abs_diff`

## legacy

- final score: `1.003436`
- final drift: x=`0.054433` size=`0.397984` vis_conf=`0.551019`
- largest FQ->ID drift point: `stage4.1.add pre-requant` (`0.391681`)
- stage4.1.add: eps_out=`0.009161373600363731` D=`32768` mul=`[49, 32768]`
- stage4.1.conv2 output: fq_vs_id mean_abs_diff=`0.318174` max_abs_diff=`1.740512`
- stage4.1 residual skip input: fq_vs_id mean_abs_diff=`0.197435` max_abs_diff=`2.235375`
- stage4.1.add pre-requant: fq_vs_id mean_abs_diff=`0.391681` max_abs_diff=`3.118182`
- stage4.1.add post-requant: fq_vs_id mean_abs_diff=`0.391059` max_abs_diff=`3.120053`
- global pool output: fq_vs_id mean_abs_diff=`0.203548` max_abs_diff=`1.012834`
- head input: fq_vs_id mean_abs_diff=`0.203548` max_abs_diff=`1.012834`

## sqrt_fanin

- final score: `1.009752`
- final drift: x=`0.057118` size=`0.399602` vis_conf=`0.553032`
- largest FQ->ID drift point: `stage4.1.add pre-requant` (`0.391817`)
- stage4.1.add: eps_out=`0.012956138700246811` D=`32768` mul=`[35, 23170]`
- stage4.1.conv2 output: fq_vs_id mean_abs_diff=`0.318279` max_abs_diff=`1.740581`
- stage4.1 residual skip input: fq_vs_id mean_abs_diff=`0.197571` max_abs_diff=`2.235375`
- stage4.1.add pre-requant: fq_vs_id mean_abs_diff=`0.391817` max_abs_diff=`3.118171`
- stage4.1.add post-requant: fq_vs_id mean_abs_diff=`0.391084` max_abs_diff=`3.125419`
- global pool output: fq_vs_id mean_abs_diff=`0.202026` max_abs_diff=`1.012834`
- head input: fq_vs_id mean_abs_diff=`0.202026` max_abs_diff=`1.012834`

## midpoint

- final score: `1.004311`
- final drift: x=`0.052998` size=`0.397588` vis_conf=`0.553725`
- largest FQ->ID drift point: `stage4.1.add pre-requant` (`0.391937`)
- stage4.1.add: eps_out=`0.011058757081627846` D=`32768` mul=`[41, 27146]`
- stage4.1.conv2 output: fq_vs_id mean_abs_diff=`0.318482` max_abs_diff=`1.740747`
- stage4.1 residual skip input: fq_vs_id mean_abs_diff=`0.197485` max_abs_diff=`2.235375`
- stage4.1.add pre-requant: fq_vs_id mean_abs_diff=`0.391937` max_abs_diff=`3.116223`
- stage4.1.add post-requant: fq_vs_id mean_abs_diff=`0.391205` max_abs_diff=`3.116258`
- global pool output: fq_vs_id mean_abs_diff=`0.204028` max_abs_diff=`1.025644`
- head input: fq_vs_id mean_abs_diff=`0.204028` max_abs_diff=`1.025644`

## fanin

- final score: `0.996848`
- final drift: x=`0.050862` size=`0.398137` vis_conf=`0.547849`
- largest FQ->ID drift point: `stage4.1.add pre-requant` (`0.391293`)
- stage4.1.add: eps_out=`0.018322747200727463` D=`65536` mul=`[49, 32768]`
- stage4.1.conv2 output: fq_vs_id mean_abs_diff=`0.318061` max_abs_diff=`1.735403`
- stage4.1 residual skip input: fq_vs_id mean_abs_diff=`0.197471` max_abs_diff=`2.235375`
- stage4.1.add pre-requant: fq_vs_id mean_abs_diff=`0.391293` max_abs_diff=`3.119607`
- stage4.1.add post-requant: fq_vs_id mean_abs_diff=`0.390068` max_abs_diff=`3.120053`
- global pool output: fq_vs_id mean_abs_diff=`0.205429` max_abs_diff=`1.012834`
- head input: fq_vs_id mean_abs_diff=`0.205429` max_abs_diff=`1.012834`
