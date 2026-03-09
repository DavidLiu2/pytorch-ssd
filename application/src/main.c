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
#include "mem.h"
#include "network.h"

#include "pmsis.h"
#include <stdint.h>

#define VERBOSE 1
#define L2_BUFFER_SIZE 412000
#define L2_INPUT_SIZE 76800
#define RAM_INPUT_SIZE 1000000
#define IO_DUMP_BYTES 64

static void dump_bytes(const char *label, const uint8_t *data, size_t data_size, size_t max_dump_size) {
  size_t dump_size = data_size < max_dump_size ? data_size : max_dump_size;
  printf("%s (%u bytes):\n", label, (unsigned int)dump_size);
  for (size_t i = 0; i < dump_size; i++) {
    if ((i % 16) == 0) {
      printf("%04u: ", (unsigned int)i);
    }
    printf("%02x ", data[i]);
    if ((i % 16) == 15 || i == dump_size - 1) {
      printf("\n");
    }
  }
}


void application(void * arg) {
/*
    Opening of Filesystem and Ram
*/
  printf("MAIN START\n");
  printf("BEFORE mem_init\n");
  mem_init();
  printf("AFTER mem_init\n");
  printf("BEFORE network_initialize\n");
  network_initialize();
  printf("AFTER network_initialize\n");
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(L2_BUFFER_SIZE);
  if (NULL == l2_buffer) {
#ifdef VERBOSE
    printf("ERROR: L2 buffer allocation failed.");
#endif
    pmsis_exit(-1);
  }
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_buffer);
#endif
  int initial_dir = 1;

  void *ram_input = ram_malloc(RAM_INPUT_SIZE);
  if (NULL == ram_input) {
    printf("ERROR: RAM input allocation failed.\n");
    pmsis_exit(-1);
  }
  printf("BEFORE load_file_to_ram\n");
  size_t loaded_input_size = load_file_to_ram(ram_input, "inputs.hex");
  printf("AFTER load_file_to_ram\n");
  printf("Loaded inputs.hex bytes: %u\n", (unsigned int)loaded_input_size);
  if (loaded_input_size < L2_INPUT_SIZE) {
    printf("ERROR: inputs.hex too small (%u < %u)\n", (unsigned int)loaded_input_size, (unsigned int)L2_INPUT_SIZE);
    pmsis_exit(-1);
  }

  printf("BEFORE ram_read\n");
  ram_read(l2_buffer, ram_input, L2_INPUT_SIZE);
  printf("AFTER ram_read\n");
  dump_bytes("INPUT_DUMP", (const uint8_t *)l2_buffer, L2_INPUT_SIZE, IO_DUMP_BYTES);

  printf("BEFORE network_run\n");
  network_run(l2_buffer, L2_BUFFER_SIZE, l2_buffer, 0, initial_dir);
  printf("AFTER network_run\n");
  dump_bytes("OUTPUT_DUMP", (const uint8_t *)l2_buffer, L2_BUFFER_SIZE, IO_DUMP_BYTES);

  ram_free(ram_input, RAM_INPUT_SIZE);
  network_terminate();
  pi_l2_free(l2_buffer, L2_BUFFER_SIZE);
}

int main () {
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


  pmsis_kickoff((void*)application);
  return 0;
}
