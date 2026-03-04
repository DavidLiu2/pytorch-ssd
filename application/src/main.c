/*
 * test_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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
#include "input.h"
#define DEFINE_CONSTANTS
#include "network.h"

#include "pmsis.h"

#define VERBOSE 1

void application(void *arg)
{
  /*
      Opening of Filesystem and Ram
  */
  /*
    Allocating space for input
  */
  printf("APPLICATION START\n");

  for (int sz = 360000; sz >= 100000; sz -= 10000) {
    void *p = pi_l2_malloc(sz);
    if (p) { printf("max-ish L2 alloc: %d @ %p\n", sz, p); pi_l2_free(p, sz); break; }
  }
  

  #define L2_ARENA_SIZE 200000

  void *l2_arena = pi_l2_malloc(L2_ARENA_SIZE);
  if (NULL == l2_arena)
  {
    #ifdef VERBOSE
        printf("ERROR: L2 buffer allocation failed.\n");
    #endif
        pmsis_exit(-1);
  }
  #ifdef VERBOSE
    printf("\nL2 Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_arena);
  #endif
  if (!l2_arena) pmsis_exit(-1);

  void *l2_out = pi_l2_malloc(1600);       // based on your “boxes” size assumption
  if (!l2_out) pmsis_exit(-2);

  // If input is already a const array in L2 (like L2_input_h), you may not need to allocate.
  // But many generated examples copy input to L2 first:
  size_t input_size = sizeof(L2_input_h);
  void *l2_in = pi_l2_malloc(input_size);
  if (!l2_in) pmsis_exit(-3);
  memcpy(l2_in, L2_input_h, input_size);
  int initial_dir = 1; 

  printf("activations_size[0] = %d\n", activations_size[0]);
  printf("sizeof(L2_input_h)  = %d\n", (int)sizeof(L2_input_h));
  printf("first byte L2_input_h[0] = %u\n", (unsigned)L2_input_h[0]);


  network_run(l2_arena, L2_ARENA_SIZE, l2_out, 0, initial_dir, l2_in);

  printf("---- OUTPUT ----\n");
  float *boxes = (float *)l2_out;

  int num_floats = 1600 / sizeof(float); // 400
  int num_boxes = num_floats / 4;        // 100

  printf("num_boxes = %d\n", num_boxes);

  for (int i = 0; i < 10; i++)
  { // print first 10 only
    float x1 = boxes[4 * i + 0];
    float y1 = boxes[4 * i + 1];
    float x2 = boxes[4 * i + 2];
    float y2 = boxes[4 * i + 3];

    printf("[%d] %f %f %f %f\n", i, x1, y1, x2, y2);
  }

  pi_l2_free(l2_arena, 200000);
}

int main()
{
#ifndef TARGET_CHIP_FAMILY_GAP9
  PMU_set_voltage(1000, 0);
#else
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, PI_PMU_VOLT_800);
#endif
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  pi_time_wait_us(10000);

  pmsis_kickoff((void *)application);
  return 0;
}
