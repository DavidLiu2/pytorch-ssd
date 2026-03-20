/*
 * Lightweight validation entry point for the current hybrid-follow GAP8 app.
 * It loads inputs.hex, runs the generated network once, and emits the final
 * 3-value int32 tensor in a log-friendly format for post-run comparison.
 */
#include "mem.h"
#include "network.h"

#include "pmsis.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef APP_INPUT_FILENAME
#define APP_INPUT_FILENAME "inputs.hex"
#endif

#ifndef APP_INPUT_BYTES
#define APP_INPUT_BYTES (128u * 128u)
#endif

#ifndef APP_L2_BUFFER_SIZE
#define APP_L2_BUFFER_SIZE 412000u
#endif

#ifndef APP_OUTPUT_COUNT
#define APP_OUTPUT_COUNT 3u
#endif

#ifndef APP_INITIAL_DIR
#define APP_INITIAL_DIR 1
#endif

#define APP_OUTPUT_BYTES (APP_OUTPUT_COUNT * sizeof(int32_t))

static void dump_final_tensor_i32(const int32_t *values, size_t count) {
  printf("FINAL_TENSOR_I32_BEGIN final count=%u\n", (unsigned int) count);
  printf("FINAL_TENSOR_I32 final");
  for (size_t i = 0; i < count; i++) {
    printf(" %ld", (long) values[i]);
  }
  printf("\n");
  printf("FINAL_TENSOR_I32_END final\n");
}

void application(void *arg) {
  (void) arg;

  mem_init();
  network_initialize();

  uint8_t *l2_buffer = (uint8_t *) pi_l2_malloc(APP_L2_BUFFER_SIZE);
  if (l2_buffer == NULL) {
    printf("ERROR: L2 allocation failed.\n");
    pmsis_exit(-1);
  }

  void *ram_input = ram_malloc(APP_INPUT_BYTES);
  if (ram_input == NULL) {
    printf("ERROR: RAM input allocation failed.\n");
    pmsis_exit(-2);
  }

  size_t input_size = load_file_to_ram(ram_input, APP_INPUT_FILENAME);
  if (input_size != APP_INPUT_BYTES) {
    printf("ERROR: %s size mismatch: expected %u got %u\n",
           APP_INPUT_FILENAME,
           (unsigned int) APP_INPUT_BYTES,
           (unsigned int) input_size);
    pmsis_exit(-3);
  }

  ram_read(l2_buffer, ram_input, APP_INPUT_BYTES);
  network_run(l2_buffer, APP_L2_BUFFER_SIZE, l2_buffer, 0, APP_INITIAL_DIR);
  dump_final_tensor_i32((const int32_t *) l2_buffer, APP_OUTPUT_COUNT);

  ram_free(ram_input, input_size);
  pi_l2_free(l2_buffer, APP_L2_BUFFER_SIZE);
  network_terminate();
}

int main(void) {
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

  pmsis_kickoff((void *) application);
  return 0;
}
