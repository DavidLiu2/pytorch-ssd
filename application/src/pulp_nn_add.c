/*
 * pulp_nn_add.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))

void __attribute__ ((noinline))  pulp_nn_add (
    uint8_t * Im_in_1,             // pointer to the input feature map1
    uint8_t * Im_in_2,             // pointer to the input feature map2
    uint8_t * Im_out,            // pointer to the output
    uint16_t out_mult1,            // paramter to requantize
    uint16_t out_mult2,            // paramter to requantize
    uint16_t out_shift,            // paramter to requantize
    uint16_t  dim_im_in_h,
    uint16_t  dim_im_in_w,
    uint16_t  ch_im_in          // number of channels of the IFM
)
{
    int core_id = pi_core_id();
    int n_cores = NUM_CORES;
    if (dim_im_in_h < NUM_CORES)
    {
      n_cores = dim_im_in_h;
    }
    int  Log2Core = log2(n_cores);

    int chunck = (dim_im_in_h >> Log2Core) + ((dim_im_in_h & (NUM_CORES-1))!=0);

    int start = min(chunck * core_id, dim_im_in_h);
    int stop = min(start + chunck, dim_im_in_h);
    uint8_t *target1 = Im_in_1 + start*ch_im_in*dim_im_in_w;
    uint8_t *target2 = Im_in_2 + start*ch_im_in*dim_im_in_w;
    uint8_t *pOut = Im_out + start*ch_im_in*dim_im_in_w;
    for (int spatial = 0; spatial<dim_im_in_w*ch_im_in*(stop-start); spatial+=1)
    {
        *pOut = pulp_nn_add_quant_u8(*target1, *target2, out_mult1, out_mult2, out_shift);
        target1 += 1;
        target2 += 1;
        pOut += 1;
    }
   pi_cl_team_barrier(0);
}

void __attribute__ ((noinline)) pulp_nn_add_raw_i32_u8(
  const int32_t *Im_in_1,
  const int32_t *Im_in_2,
  uint8_t *Im_out,
  uint16_t in_mult1,
  uint16_t in_mult2,
  uint16_t add_shift,
  uint16_t out_mult,
  uint16_t out_shift,
  uint32_t num_elements,
  const char *layer_name
) {
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  (void) layer_name;
  if ((uint32_t) n_cores > num_elements && num_elements > 0) {
    n_cores = (int) num_elements;
  }
  if (n_cores <= 0) {
    n_cores = 1;
  }

  int log2_core = log2(n_cores);
  uint32_t chunk = (num_elements >> log2_core) + ((num_elements & (n_cores - 1)) != 0);
  uint32_t start = min(chunk * core_id, num_elements);
  uint32_t stop = min(start + chunk, num_elements);

  for (uint32_t i = start; i < stop; i++) {
    int32_t acc = (int32_t) ((((int64_t) Im_in_1[i] * in_mult1) + ((int64_t) Im_in_2[i] * in_mult2)) >> add_shift);
    Im_out[i] = pulp_nn_quant_u8(acc, (int16_t) out_mult, (int8_t) out_shift);
  }

  pi_cl_team_barrier(0);
}

void __attribute__ ((noinline)) pulp_nn_add_raw_i32_u8_mixed(
  const int32_t *Im_in_1,
  const uint8_t *Im_in_2,
  uint8_t *Im_out,
  uint16_t in_mult1,
  uint16_t in_mult2,
  uint16_t add_shift,
  uint16_t out_mult,
  uint16_t out_shift,
  uint32_t num_elements,
  const char *layer_name
) {
  int core_id = pi_core_id();
  int n_cores = NUM_CORES;
  (void) layer_name;
  if ((uint32_t) n_cores > num_elements && num_elements > 0) {
    n_cores = (int) num_elements;
  }
  if (n_cores <= 0) {
    n_cores = 1;
  }

  int log2_core = log2(n_cores);
  uint32_t chunk = (num_elements >> log2_core) + ((num_elements & (n_cores - 1)) != 0);
  uint32_t start = min(chunk * core_id, num_elements);
  uint32_t stop = min(start + chunk, num_elements);

  for (uint32_t i = start; i < stop; i++) {
    int32_t acc = (int32_t) ((((int64_t) Im_in_1[i] * in_mult1) + ((int64_t) Im_in_2[i] * in_mult2)) >> add_shift);
    Im_out[i] = pulp_nn_quant_u8(acc, (int16_t) out_mult, (int8_t) out_shift);
  }

  pi_cl_team_barrier(0);
}
