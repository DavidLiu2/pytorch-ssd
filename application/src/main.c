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
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifndef APP_PROGRESS_ONLY_MODE
#define APP_PROGRESS_ONLY_MODE 1
#endif

#ifndef APP_DEBUG_INPUT_SUMMARY
#define APP_DEBUG_INPUT_SUMMARY (!APP_PROGRESS_ONLY_MODE)
#endif

#ifndef APP_DEBUG_FINAL_SUMMARY
#define APP_DEBUG_FINAL_SUMMARY 0
#endif

#ifndef APP_DEBUG_FULL_FINAL_I32_DUMP
#define APP_DEBUG_FULL_FINAL_I32_DUMP 0
#endif

#ifndef APP_PROGRESS_POLL_US
#define APP_PROGRESS_POLL_US 1000
#endif

#ifndef APP_PROGRESS_FILE
#define APP_PROGRESS_FILE "network_progress.txt"
#endif

#ifndef APP_BBOX_DUMP_FILE
#define APP_BBOX_DUMP_FILE "bbox.bin"
#endif

#ifndef APP_CLS_DUMP_FILE
#define APP_CLS_DUMP_FILE "cls.bin"
#endif

#ifndef APP_L2_BUFFER_SIZE
// The current schedule peaks at ~328 KB of L2 scratch, so keep a small margin
// instead of requesting the old 412 KB block that no longer fits FC-side runtime L2.
#define APP_L2_BUFFER_SIZE 344064
#endif

#ifndef APP_PROGRESS_STDOUT
#define APP_PROGRESS_STDOUT 0
#endif

static char g_progress_file_buffer[1024];

static void print_u8_preview(const char *label, const uint8_t *data, size_t size, size_t preview_count) {
  size_t limit = preview_count < size ? preview_count : size;
  printf("%s (%u bytes):", label, (unsigned int)limit);
  for (size_t i = 0; i < limit; i++) {
    printf(" %u", (unsigned int)data[i]);
  }
  printf("\n");
}

static void print_u8_stats(const char *label, const uint8_t *data, size_t size) {
  uint32_t sum = 0;
  uint8_t min_v = 255;
  uint8_t max_v = 0;
  for (size_t i = 0; i < size; i++) {
    uint8_t v = data[i];
    sum += v;
    if (v < min_v) min_v = v;
    if (v > max_v) max_v = v;
  }
  printf("%s stats: size=%u sum=%u min=%u max=%u\n",
         label, (unsigned int)size, (unsigned int)sum, (unsigned int)min_v, (unsigned int)max_v);
}

static void print_ram_u8_preview(const char *label, void *ram_src, size_t size, size_t preview_count) {
  uint8_t preview[128];
  size_t limit = preview_count < size ? preview_count : size;
  if (limit > sizeof(preview)) {
    limit = sizeof(preview);
  }
  if (limit > 0) {
    ram_read(preview, ram_src, limit);
  }
  print_u8_preview(label, preview, limit, limit);
}

static void print_ram_u8_stats(const char *label, void *ram_src, size_t size) {
  uint8_t scratch[256];
  uint64_t sum = 0;
  uint8_t min_v = 255;
  uint8_t max_v = 0;

  for (size_t offset = 0; offset < size; offset += sizeof(scratch)) {
    size_t chunk = sizeof(scratch);
    if (chunk > (size - offset)) {
      chunk = size - offset;
    }
    ram_read(scratch, (uint8_t *) ram_src + offset, chunk);
    for (size_t i = 0; i < chunk; i++) {
      uint8_t v = scratch[i];
      sum += v;
      if (v < min_v) min_v = v;
      if (v > max_v) max_v = v;
    }
  }

  printf("%s byte-stats: size=%u sum=%u min=%u max=%u\n",
         label,
         (unsigned int) size,
         (unsigned int) sum,
         (unsigned int) min_v,
         (unsigned int) max_v);
}

static void print_ram_i32_stats(const char *label, void *ram_src, size_t size_bytes) {
  int32_t scratch[64];
  size_t count = size_bytes / sizeof(int32_t);
  int64_t sum = 0;
  int32_t min_v = 0;
  int32_t max_v = 0;
  int initialized = 0;

  for (size_t offset = 0; offset < count; offset += 64) {
    size_t chunk_elems = 64;
    if (chunk_elems > (count - offset)) {
      chunk_elems = count - offset;
    }
    ram_read(scratch, (uint8_t *) ram_src + offset * sizeof(int32_t), chunk_elems * sizeof(int32_t));
    for (size_t i = 0; i < chunk_elems; i++) {
      int32_t v = scratch[i];
      sum += v;
      if (!initialized || v < min_v) min_v = v;
      if (!initialized || v > max_v) max_v = v;
      initialized = 1;
    }
  }

  printf("%s int32-stats: count=%u sum=%lld min=%d max=%d\n",
         label,
         (unsigned int) count,
         (long long) sum,
         (int) min_v,
         (int) max_v);
}

static void dump_ram_i32_tensor(const char *label, void *ram_src, size_t size_bytes) {
  int32_t scratch[64];
  size_t count = size_bytes / sizeof(int32_t);

  printf("FINAL_TENSOR_I32_BEGIN %s count=%u\n", label, (unsigned int) count);
  for (size_t offset = 0; offset < count; offset += 64) {
    size_t chunk_elems = 64;
    if (chunk_elems > (count - offset)) {
      chunk_elems = count - offset;
    }
    ram_read(scratch, (uint8_t *) ram_src + offset * sizeof(int32_t), chunk_elems * sizeof(int32_t));
    printf("FINAL_TENSOR_I32 %s", label);
    for (size_t i = 0; i < chunk_elems; i++) {
      printf(" %d", (int) scratch[i]);
    }
    printf("\n");
  }
  printf("FINAL_TENSOR_I32_END %s\n", label);
}

static void report_final_tensor(const char *label, void *ram_src, size_t size_bytes) {
  printf("Final tensor %s\n", label);
  print_ram_u8_stats(label, ram_src, size_bytes);
  print_ram_i32_stats(label, ram_src, size_bytes);
  print_ram_u8_preview("Final first 64 bytes", ram_src, size_bytes, 64);
  print_ram_u8_preview("Final first 128 bytes", ram_src, size_bytes, 128);
#if APP_DEBUG_FULL_FINAL_I32_DUMP
  dump_ram_i32_tensor(label, ram_src, size_bytes);
#endif
}

static void emit_progress_stdout(const struct network_progress_state *progress) {
#if APP_PROGRESS_STDOUT
  printf(
      "PROGRESS layer=%d name=%s marker_epoch=%u marker_layer=%d marker=%s error=%s heartbeat=%u sample_sum=%u cycles=%u bbox_bytes=%u cls_bytes=%u finished=%u status=%u\n",
      (int) progress->latest_completed_layer,
      network_get_layer_name(progress->latest_completed_layer),
      (unsigned int) progress->marker_epoch,
      (int) progress->marker_layer,
      progress->latest_marker,
      progress->latest_error,
      (unsigned int) progress->heartbeat_count,
      (unsigned int) progress->latest_output_sample_sum,
      (unsigned int) progress->cycle_count,
      (unsigned int) progress->bbox_bytes_written,
      (unsigned int) progress->cls_bytes_written,
      (unsigned int) progress->finished,
      (unsigned int) progress->final_status);
#else
  (void) progress;
#endif
}

static int write_progress_file(const struct network_progress_state *progress, uint32_t target_time_us) {
  int length = snprintf(
      g_progress_file_buffer,
      sizeof(g_progress_file_buffer),
      "st=%s\n"
      "sd=%u\n"
      "fn=%u\n"
      "l=%d\n"
      "dr=%d\n"
      "ln=%s\n"
      "hb=%u\n"
      "cy=%u\n"
      "bb=%u\n"
      "cb=%u\n"
      "fs=%u\n"
      "me=%u\n"
      "ml=%d\n"
      "mn=%s\n"
      "er=%s\n"
      "ab=0x%08x\n"
      "ae=0x%08x\n"
      "ib=0x%08x\n"
      "is=%u\n"
      "ob=0x%08x\n"
      "os=%u\n"
      "wb=0x%08x\n"
      "ws=%u\n"
      "bo=0x%08x\n"
      "co=0x%08x\n"
      "cd=0x%08x\n"
      "cz=%u\n"
      "alc=0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,%u,%u,%d,%d\n"
      "p66=%u,%d,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,%u,%u,%d,%d\n"
      "p67=%u,%d,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,%u,%u,%d,%d\n"
      "tt=%u\n",
      progress->finished ? "finished" : "running",
      (unsigned int) progress->started,
      (unsigned int) progress->finished,
      (int) progress->latest_completed_layer,
      (int) progress->current_direction,
      network_get_layer_name(progress->latest_completed_layer),
      (unsigned int) progress->heartbeat_count,
      (unsigned int) progress->cycle_count,
      (unsigned int) progress->bbox_bytes_written,
      (unsigned int) progress->cls_bytes_written,
      (unsigned int) progress->final_status,
      (unsigned int) progress->marker_epoch,
      (int) progress->marker_layer,
      progress->latest_marker,
      progress->latest_error,
      (unsigned int) progress->allocator_begin_addr,
      (unsigned int) progress->allocator_end_addr,
      (unsigned int) progress->l2_input_addr,
      (unsigned int) progress->l2_input_size,
      (unsigned int) progress->l2_output_addr,
      (unsigned int) progress->l2_output_size,
      (unsigned int) progress->l2_weights_addr,
      (unsigned int) progress->l2_weights_size,
      (unsigned int) progress->bbox_output_addr,
      (unsigned int) progress->cls_output_addr,
      (unsigned int) progress->capture_dest_addr,
      (unsigned int) progress->capture_size,
      (unsigned int) progress->allocator.base_addr,
      (unsigned int) progress->allocator.limit_addr,
      (unsigned int) progress->allocator.last_before_begin_addr,
      (unsigned int) progress->allocator.last_before_end_addr,
      (unsigned int) progress->allocator.last_after_begin_addr,
      (unsigned int) progress->allocator.last_after_end_addr,
      (unsigned int) progress->allocator.last_return_addr,
      (unsigned int) progress->allocator.last_request_size,
      (unsigned int) progress->allocator.last_alignment,
      (int) progress->allocator.last_direction,
      (int) progress->allocator.last_operation,
      (unsigned int) progress->layer66_output_alloc.valid,
      (int) progress->layer66_output_alloc.layer_id,
      (unsigned int) progress->layer66_output_alloc.allocator.base_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.limit_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.last_before_begin_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.last_before_end_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.last_after_begin_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.last_after_end_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.last_return_addr,
      (unsigned int) progress->layer66_output_alloc.allocator.last_request_size,
      (unsigned int) progress->layer66_output_alloc.allocator.last_alignment,
      (int) progress->layer66_output_alloc.allocator.last_direction,
      (int) progress->layer66_output_alloc.allocator.last_operation,
      (unsigned int) progress->layer67_output_alloc.valid,
      (int) progress->layer67_output_alloc.layer_id,
      (unsigned int) progress->layer67_output_alloc.allocator.base_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.limit_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.last_before_begin_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.last_before_end_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.last_after_begin_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.last_after_end_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.last_return_addr,
      (unsigned int) progress->layer67_output_alloc.allocator.last_request_size,
      (unsigned int) progress->layer67_output_alloc.allocator.last_alignment,
      (int) progress->layer67_output_alloc.allocator.last_direction,
      (int) progress->layer67_output_alloc.allocator.last_operation,
      (unsigned int) target_time_us);

  if (length < 0) {
    printf("ERROR: Could not format progress file.\n");
    return -1;
  }

  if ((size_t) length >= sizeof(g_progress_file_buffer)) {
    length = (int) sizeof(g_progress_file_buffer) - 1;
  }

  return hostfs_write_file(APP_PROGRESS_FILE, g_progress_file_buffer, (size_t) length);
}

static void export_final_tensor_file(const char *label, const char *filename, void *ram_src, size_t size_bytes) {
  int written = hostfs_write_file_from_ram(filename, ram_src, size_bytes);
  if (written < 0) {
    printf("ERROR: Failed to export %s tensor to %s (err=%d)\n", label, filename, written);
  } else {
    printf("Exported %s tensor to %s (%u bytes)\n", label, filename, (unsigned int) size_bytes);
  }
}

static void monitor_network_progress(struct network_run_token token) {
  struct network_progress_state progress = {0};
  int32_t last_completed_layer = -2;
  uint32_t last_heartbeat = 0xffffffffu;
  uint32_t last_finished = 0xffffffffu;
  uint32_t last_final_status = 0xffffffffu;
  uint32_t last_marker_epoch = 0xffffffffu;

  while (1) {
    if (network_get_progress_snapshot(&progress) != 0) {
      printf("ERROR: Failed to read network progress snapshot.\n");
      break;
    }

    if ((progress.latest_completed_layer != last_completed_layer) ||
        (progress.heartbeat_count != last_heartbeat) ||
        (progress.finished != last_finished) ||
        (progress.final_status != last_final_status) ||
        (progress.marker_epoch != last_marker_epoch)) {
      write_progress_file(&progress, pi_time_get_us());
      emit_progress_stdout(&progress);
      last_completed_layer = progress.latest_completed_layer;
      last_heartbeat = progress.heartbeat_count;
      last_finished = progress.finished;
      last_final_status = progress.final_status;
      last_marker_epoch = progress.marker_epoch;
    }

    if (progress.finished) {
      break;
    }

    pi_time_wait_us(APP_PROGRESS_POLL_US);
  }

  network_run_wait(token);
  if (network_get_progress_snapshot(&progress) == 0) {
    write_progress_file(&progress, pi_time_get_us());
    emit_progress_stdout(&progress);
  }
}


void application(void * arg) {
/*
    Opening of Filesystem and Ram
*/
  mem_init();
  network_initialize();
  /*
    Allocating space for input
  */
  void *l2_buffer = pi_l2_malloc(APP_L2_BUFFER_SIZE);
  size_t l2_input_size = 16384;
  size_t input_size = l2_input_size;
  void *bbox_output = ram_malloc(NETWORK_BBOX_OUTPUT_SIZE_BYTES);
  void *cls_output = ram_malloc(NETWORK_CLS_OUTPUT_SIZE_BYTES);
  if (NULL == l2_buffer) {
    printf("ERROR: L2 buffer allocation failed.\n");
    pmsis_exit(-1);
  }
  if (bbox_output == NULL || cls_output == NULL) {
    printf("ERROR: Final output buffer allocation failed.\n");
    pmsis_exit(-6);
  }
#if APP_DEBUG_INPUT_SUMMARY
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\tOk\n", (unsigned int)l2_buffer);
#endif
  int initial_dir = 1;

  void *ram_input = ram_malloc(input_size);
  if (ram_input == NULL) {
    printf("ERROR: Input staging buffer allocation failed for %u bytes.\n", (unsigned int) input_size);
    pmsis_exit(-8);
  }
  load_file_to_ram(ram_input, "inputs.hex");
  ram_read(l2_buffer, ram_input, l2_input_size);
#if APP_DEBUG_INPUT_SUMMARY
  print_u8_stats("Input buffer before network_run", (const uint8_t *)l2_buffer, l2_input_size);
  print_u8_preview("Input first 32", (const uint8_t *)l2_buffer, l2_input_size, 32);
  print_u8_preview("Input first 128", (const uint8_t *)l2_buffer, l2_input_size, 128);
#endif
  {
    const struct network_final_outputs final_outputs = {
      .bbox_output = bbox_output,
      .cls_output = cls_output,
    };
    struct network_run_token token = network_run_outputs_async(l2_buffer, APP_L2_BUFFER_SIZE, NULL, final_outputs, 0, initial_dir);
    monitor_network_progress(token);
  }

  export_final_tensor_file("bbox", APP_BBOX_DUMP_FILE, bbox_output, NETWORK_BBOX_OUTPUT_SIZE_BYTES);
  export_final_tensor_file("cls", APP_CLS_DUMP_FILE, cls_output, NETWORK_CLS_OUTPUT_SIZE_BYTES);

#if APP_DEBUG_FINAL_SUMMARY
  report_final_tensor("bbox", bbox_output, NETWORK_BBOX_OUTPUT_SIZE_BYTES);
  report_final_tensor("cls", cls_output, NETWORK_CLS_OUTPUT_SIZE_BYTES);
#endif

  ram_free(ram_input, input_size);
  ram_free(bbox_output, NETWORK_BBOX_OUTPUT_SIZE_BYTES);
  ram_free(cls_output, NETWORK_CLS_OUTPUT_SIZE_BYTES);
  network_terminate();
  pi_l2_free(l2_buffer, APP_L2_BUFFER_SIZE);
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
