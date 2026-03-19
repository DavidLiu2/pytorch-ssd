/*
 * network.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
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
#define DEFINE_CONSTANTS
#include "net_utils.h"
#include "pmsis.h"
#include "network.h"
#include "directional_allocator.h"
#include "mem.h"
#include <string.h>
#include "ReluConvolution0.h"
#include "Convolution27.h"
#include "ReluConvolution29.h"
#include "Addition47.h"
#include "Convolution16.h"
#include "ReluConvolution18.h"
#include "Addition3.h"
#include "Convolution6.h"
#include "ReluConvolution60.h"
#include "ReluConvolution51.h"
#include "Addition21.h"
#include "Convolution46.h"
#include "Convolution67.h"
#include "ReluConvolution15.h"
#include "ReluConvolution56.h"
#include "Convolution57.h"
#include "Convolution68.h"
#include "Addition36.h"
#include "ReluConvolution8.h"
#include "ReluConvolution44.h"
#include "ReluConvolution59.h"
#include "ReluConvolution5.h"
#include "Convolution61.h"
#include "Addition32.h"
#include "Convolution2.h"
#include "Convolution69.h"
#include "Convolution66.h"
#include "Convolution39.h"
#include "ReluConvolution55.h"
#include "Convolution63.h"
#include "ReluConvolution26.h"
#include "Convolution53.h"
#include "Convolution42.h"
#include "ReluConvolution11.h"
#include "Convolution20.h"
#include "ReluConvolution7.h"
#include "ReluConvolution30.h"
#include "ReluConvolution12.h"
#include "Addition17.h"
#include "Convolution13.h"
#include "ReluConvolution23.h"
#include "ReluConvolution45.h"
#include "Convolution50.h"
#include "Addition43.h"
#include "ReluConvolution52.h"
#include "Convolution9.h"
#include "ReluConvolution4.h"
#include "Convolution24.h"
#include "Addition28.h"
#include "ReluConvolution34.h"
#include "ReluConvolution40.h"
#include "Addition54.h"
#include "ReluConvolution25.h"
#include "ReluConvolution37.h"
#include "Convolution64.h"
#include "Addition10.h"
#include "ReluConvolution62.h"
#include "Convolution65.h"
#include "ReluConvolution19.h"
#include "Convolution31.h"
#include "ReluConvolution33.h"
#include "ReluConvolution49.h"
#include "Convolution70.h"
#include "ReluConvolution41.h"
#include "ReluConvolution1.h"
#include "ReluConvolution48.h"
#include "Convolution35.h"
#include "Addition58.h"
#include "ReluConvolution38.h"
#include "ReluConvolution22.h"
#include "ReluConvolution14.h"

#ifndef APP_PROGRESS_ONLY_MODE
#define APP_PROGRESS_ONLY_MODE 1
#endif

#ifndef NET_DEBUG_STARTUP
#define NET_DEBUG_STARTUP 0
#endif

#ifndef NET_DEBUG_LAYER0_SUMMARY
#define NET_DEBUG_LAYER0_SUMMARY (!APP_PROGRESS_ONLY_MODE)
#endif

#ifndef NET_DEBUG_LAYER_CHECKSUMS
#define NET_DEBUG_LAYER_CHECKSUMS 0
#endif

#ifndef NET_DEBUG_PROGRESS_INTERVAL
#define NET_DEBUG_PROGRESS_INTERVAL 0
#endif

#ifndef NET_PROGRESS_HEARTBEAT_INTERVAL
#define NET_PROGRESS_HEARTBEAT_INTERVAL 5
#endif

#ifndef NET_PROGRESS_SAMPLE_BYTES
#define NET_PROGRESS_SAMPLE_BYTES 128
#endif

#define NETWORK_BBOX_FIRST_LAYER 63
#define NETWORK_BBOX_LAST_LAYER 66
#define NETWORK_CLS_FIRST_LAYER 67
#define NETWORK_CLS_LAST_LAYER 70
#define NETWORK_PROGRESS_STATUS_RUNNING 0
#define NETWORK_PROGRESS_STATUS_SUCCESS 1
#define NETWORK_PROGRESS_STATUS_CAPTURE_OVERFLOW 2
#define NETWORK_PROGRESS_STATUS_NULL_POINTER 3
#define NETWORK_PROGRESS_STATUS_POINTER_RANGE 4
#define NETWORK_PROGRESS_STATUS_CAPTURE_NULL_DEST 5
#define NETWORK_PROGRESS_STATUS_ALLOC_FAILURE 6
#define NETWORK_PROGRESS_STATUS_HEAD_CACHE_ERROR 7

#define NETWORK_HEAD_SOURCE_COUNT 4
static const int g_network_head_source_layers[NETWORK_HEAD_SOURCE_COUNT] = {17, 36, 58, 62};

#define L3_WEIGHTS_SIZE 4000000
#define L3_INPUT_SIZE 1500000
#define L3_OUTPUT_SIZE 1500000
static void *L3_weights = NULL;
static void *L3_input = NULL;
static void *L3_output = NULL;
int cycle_network_execution;

typedef struct {
  unsigned int l2_buffer;
  unsigned int l2_buffer_size;
  unsigned int l2_final_output;
  unsigned int bbox_output;
  unsigned int cls_output;
  unsigned int exec;
  unsigned int initial_dir;
} network_run_cluster_args_t;

typedef struct {
  struct pi_device cluster_dev;
  struct pi_cluster_task cluster_task;
  pi_task_t done_task;
  network_run_cluster_args_t args;
  int active;
} network_async_state_t;

typedef struct {
  uint32_t l2_buffer_base_addr;
  uint32_t l2_buffer_total_size;
  uint32_t current_direction;
  uint32_t bbox_output_addr;
  uint32_t cls_output_addr;
  uint32_t bbox_offset;
  uint32_t cls_offset;
} network_progress_context_t;

static network_async_state_t g_network_async_state = {0};
static volatile network_progress_context_t g_network_progress_ctx = {0};
static volatile struct network_progress_state g_network_progress = {
  .latest_completed_layer = -1,
  .marker_layer = -1,
};

static void progress_begin_update(void);
static void progress_end_update(void);

static void print_u8_summary(const char *label, const uint8_t *data, size_t size, size_t preview_count) {
  uint32_t sum = 0;
  uint8_t min_v = 255;
  uint8_t max_v = 0;
  size_t limit = preview_count < size ? preview_count : size;
  for (size_t i = 0; i < size; i++) {
    uint8_t v = data[i];
    sum += v;
    if (v < min_v) min_v = v;
    if (v > max_v) max_v = v;
  }

  printf("%s summary: size=%u sum=%u min=%u max=%u\n",
         label, (unsigned int) size, (unsigned int) sum, (unsigned int) min_v, (unsigned int) max_v);
  printf("%s first %u:", label, (unsigned int) limit);
  for (size_t i = 0; i < limit; i++) {
    printf(" %u", (unsigned int) data[i]);
  }
  printf("\n");
}

static uint32_t sample_u8_sum(const uint8_t *data, size_t size, size_t sample_limit) {
  size_t limit = sample_limit < size ? sample_limit : size;
  uint32_t sum = 0;

  for (size_t i = 0; i < limit; i++) {
    sum += data[i];
  }

  return sum;
}

static void progress_copy_text(char *dst, size_t dst_size, const char *src) {
  size_t idx = 0;

  if (dst_size == 0) {
    return;
  }

  if (src == NULL) {
    dst[0] = '\0';
    return;
  }

  while ((idx + 1) < dst_size && src[idx] != '\0') {
    dst[idx] = src[idx];
    idx++;
  }
  dst[idx] = '\0';
}

static void progress_store_allocator_state(
    volatile struct network_allocator_state *dst,
    const directional_allocator_debug_state_t *src) {
  if (dst == NULL || src == NULL) {
    return;
  }

  dst->base_addr = (uint32_t) src->base_addr;
  dst->limit_addr = (uint32_t) src->limit_addr;
  dst->begin_addr = (uint32_t) src->begin_addr;
  dst->end_addr = (uint32_t) src->end_addr;
  dst->last_before_begin_addr = (uint32_t) src->last_before_begin_addr;
  dst->last_before_end_addr = (uint32_t) src->last_before_end_addr;
  dst->last_after_begin_addr = (uint32_t) src->last_after_begin_addr;
  dst->last_after_end_addr = (uint32_t) src->last_after_end_addr;
  dst->last_return_addr = (uint32_t) src->last_return_addr;
  dst->last_request_size = (uint32_t) src->last_request_size;
  dst->last_alignment = src->last_alignment;
  dst->last_direction = src->last_direction;
  dst->last_operation = src->last_operation;
}

static void progress_clear_allocator_state(volatile struct network_allocator_state *state) {
  if (state == NULL) {
    return;
  }

  memset((void *) state, 0, sizeof(*state));
}

static void progress_clear_output_alloc_probe(volatile struct network_output_alloc_probe *probe) {
  if (probe == NULL) {
    return;
  }

  probe->valid = 0;
  probe->layer_id = -1;
  progress_clear_allocator_state(&probe->allocator);
}

static volatile struct network_output_alloc_probe *progress_output_alloc_probe_for_layer(int layer_id) {
  if (layer_id == 66) {
    return &g_network_progress.layer66_output_alloc;
  }
  if (layer_id == 67) {
    return &g_network_progress.layer67_output_alloc;
  }
  return NULL;
}

static void progress_record_output_alloc_probe(int layer_id) {
  directional_allocator_debug_state_t allocator_state = {0};
  volatile struct network_output_alloc_probe *probe = progress_output_alloc_probe_for_layer(layer_id);

  if (probe == NULL) {
    return;
  }

  directional_allocator_get_debug_state(&allocator_state);
  progress_begin_update();
  probe->valid = 1;
  probe->layer_id = layer_id;
  progress_store_allocator_state(&probe->allocator, &allocator_state);
  progress_end_update();
}

static int network_head_cache_index_from_source_layer(int layer_id) {
  for (int idx = 0; idx < NETWORK_HEAD_SOURCE_COUNT; idx++) {
    if (g_network_head_source_layers[idx] == layer_id) {
      return idx;
    }
  }

  return -1;
}

static int network_head_cache_index_for_layer(int layer_id) {
  if (layer_id < NETWORK_BBOX_FIRST_LAYER || layer_id > NETWORK_CLS_LAST_LAYER) {
    return -1;
  }

  return (layer_id - NETWORK_BBOX_FIRST_LAYER) % NETWORK_HEAD_SOURCE_COUNT;
}

static int network_layer_uses_head_cache_input(int layer_id) {
  return network_head_cache_index_for_layer(layer_id) >= 0;
}

static int ptr_region_fits(const void *ptr, size_t region_size, const void *base, size_t total_size) {
  uintptr_t ptr_addr = (uintptr_t) ptr;
  uintptr_t base_addr = (uintptr_t) base;

  if (ptr == NULL) {
    return 0;
  }

  if (ptr_addr < base_addr) {
    return 0;
  }

  if (region_size > total_size) {
    return 0;
  }

  return (ptr_addr - base_addr) <= (total_size - region_size);
}

static void progress_clear_marker_fields(void) {
  g_network_progress.marker_layer = -1;
  g_network_progress.l2_buffer_base_addr = 0;
  g_network_progress.l2_buffer_total_size = 0;
  g_network_progress.current_direction = 0;
  progress_clear_allocator_state(&g_network_progress.allocator);
  g_network_progress.allocator_begin_addr = 0;
  g_network_progress.allocator_end_addr = 0;
  g_network_progress.l2_input_addr = 0;
  g_network_progress.l2_input_size = 0;
  g_network_progress.l2_output_addr = 0;
  g_network_progress.l2_output_size = 0;
  g_network_progress.l2_weights_addr = 0;
  g_network_progress.l2_weights_size = 0;
  g_network_progress.l3_copy_src_addr = 0;
  g_network_progress.l3_copy_dst_addr = 0;
  g_network_progress.l3_copy_size = 0;
  g_network_progress.bbox_output_addr = 0;
  g_network_progress.cls_output_addr = 0;
  g_network_progress.capture_dest_addr = 0;
  g_network_progress.capture_size = 0;
  g_network_progress.latest_marker[0] = '\0';
  g_network_progress.latest_error[0] = '\0';
}

void network_note_runtime_marker(
    const char *marker,
    int layer,
    const void *l2_input,
    size_t l2_input_size,
    const void *l2_output,
    size_t l2_output_size,
    const void *l2_weights,
    size_t l2_weights_size,
    const void *l3_copy_src,
    const void *l3_copy_dst,
    size_t l3_copy_size,
    const void *capture_dest,
    size_t capture_size,
    const char *error_text) {
  directional_allocator_debug_state_t allocator_state = {0};

  directional_allocator_get_debug_state(&allocator_state);
  progress_begin_update();
  g_network_progress.marker_epoch++;
  g_network_progress.marker_layer = layer;
  g_network_progress.l2_buffer_base_addr = g_network_progress_ctx.l2_buffer_base_addr;
  g_network_progress.l2_buffer_total_size = g_network_progress_ctx.l2_buffer_total_size;
  g_network_progress.current_direction = (int32_t) g_network_progress_ctx.current_direction;
  progress_store_allocator_state(&g_network_progress.allocator, &allocator_state);
  g_network_progress.allocator_begin_addr = g_network_progress.allocator.begin_addr;
  g_network_progress.allocator_end_addr = g_network_progress.allocator.end_addr;
  g_network_progress.l2_input_addr = (uint32_t) l2_input;
  g_network_progress.l2_input_size = (uint32_t) l2_input_size;
  g_network_progress.l2_output_addr = (uint32_t) l2_output;
  g_network_progress.l2_output_size = (uint32_t) l2_output_size;
  g_network_progress.l2_weights_addr = (uint32_t) l2_weights;
  g_network_progress.l2_weights_size = (uint32_t) l2_weights_size;
  g_network_progress.l3_copy_src_addr = (uint32_t) l3_copy_src;
  g_network_progress.l3_copy_dst_addr = (uint32_t) l3_copy_dst;
  g_network_progress.l3_copy_size = (uint32_t) l3_copy_size;
  g_network_progress.bbox_output_addr = g_network_progress_ctx.bbox_output_addr;
  g_network_progress.cls_output_addr = g_network_progress_ctx.cls_output_addr;
  g_network_progress.capture_dest_addr = (uint32_t) capture_dest;
  g_network_progress.capture_size = (uint32_t) capture_size;
  progress_copy_text(g_network_progress.latest_marker, sizeof(g_network_progress.latest_marker), marker);
  progress_copy_text(g_network_progress.latest_error, sizeof(g_network_progress.latest_error), error_text);
  progress_end_update();
}

static void progress_begin_update() {
  g_network_progress.sequence++;
  asm volatile("": : :"memory");
}

static void progress_end_update() {
  asm volatile("": : :"memory");
  g_network_progress.sequence++;
}

static void progress_reset() {
  progress_begin_update();
  g_network_progress.started = 1;
  g_network_progress.finished = 0;
  g_network_progress.latest_completed_layer = -1;
  g_network_progress.current_direction = 0;
  g_network_progress.heartbeat_count = 0;
  g_network_progress.latest_output_size = 0;
  g_network_progress.latest_output_sample_size = 0;
  g_network_progress.latest_output_sample_sum = 0;
  g_network_progress.cycle_count = 0;
  g_network_progress.bbox_bytes_written = 0;
  g_network_progress.cls_bytes_written = 0;
  g_network_progress.final_status = NETWORK_PROGRESS_STATUS_RUNNING;
  g_network_progress.marker_epoch = 0;
  progress_clear_output_alloc_probe(&g_network_progress.layer66_output_alloc);
  progress_clear_output_alloc_probe(&g_network_progress.layer67_output_alloc);
  progress_clear_marker_fields();
  progress_end_update();
}

static void progress_publish(
    int latest_completed_layer,
    const uint8_t *output,
    size_t output_size,
    uint32_t cycle_count,
    size_t bbox_offset,
    size_t cls_offset,
    int increment_heartbeat,
    uint32_t final_status,
    uint32_t finished) {
  progress_begin_update();
  g_network_progress.latest_completed_layer = latest_completed_layer;
  g_network_progress.latest_output_size = (uint32_t) output_size;
  g_network_progress.latest_output_sample_size =
      (uint32_t) ((output_size < NET_PROGRESS_SAMPLE_BYTES) ? output_size : NET_PROGRESS_SAMPLE_BYTES);
  g_network_progress.latest_output_sample_sum =
      output == NULL ? 0 : sample_u8_sum(output, output_size, NET_PROGRESS_SAMPLE_BYTES);
  g_network_progress.cycle_count = cycle_count;
  g_network_progress.bbox_bytes_written = (uint32_t) bbox_offset;
  g_network_progress.cls_bytes_written = (uint32_t) cls_offset;
  if (increment_heartbeat) {
    g_network_progress.heartbeat_count++;
  }
  g_network_progress.final_status = final_status;
  g_network_progress.finished = finished;
  progress_end_update();
}

static void progress_finish_with_error(
    uint32_t status_code,
    const char *marker,
    int layer,
    const void *l2_input,
    size_t l2_input_size,
    const void *l2_output,
    size_t l2_output_size,
    const void *l2_weights,
    size_t l2_weights_size,
    const void *l3_copy_src,
    const void *l3_copy_dst,
    size_t l3_copy_size,
    const void *capture_dest,
    size_t capture_size,
    const char *error_text) {
  network_note_runtime_marker(
      marker,
      layer,
      l2_input,
      l2_input_size,
      l2_output,
      l2_output_size,
      l2_weights,
      l2_weights_size,
      l3_copy_src,
      l3_copy_dst,
      l3_copy_size,
      capture_dest,
      capture_size,
      error_text);

  progress_begin_update();
  g_network_progress.final_status = status_code;
  g_network_progress.finished = 1;
  progress_end_update();
}

static int validate_layer_l2_pointers(
    int layer_id,
    const void *l2_buffer,
    size_t l2_buffer_size,
    const void *l2_input,
    size_t l2_input_size,
    const void *l2_output,
    size_t l2_output_size,
    const void *l2_weights,
    size_t l2_weights_size) {
  if (l2_output == NULL) {
    progress_finish_with_error(
        NETWORK_PROGRESS_STATUS_NULL_POINTER,
        "PTR_ERROR",
        layer_id,
        l2_input,
        l2_input_size,
        l2_output,
        l2_output_size,
        l2_weights,
        l2_weights_size,
        NULL,
        NULL,
        0,
        NULL,
        0,
        "L2_output is NULL");
    return -1;
  }

  if (!ptr_region_fits(l2_output, l2_output_size, l2_buffer, l2_buffer_size)) {
    progress_finish_with_error(
        NETWORK_PROGRESS_STATUS_POINTER_RANGE,
        "PTR_ERROR",
        layer_id,
        l2_input,
        l2_input_size,
        l2_output,
        l2_output_size,
        l2_weights,
        l2_weights_size,
        NULL,
        NULL,
        0,
        NULL,
        0,
        "L2_output outside scratch buffer");
    return -1;
  }

  if (l2_input_size > 0) {
    if (l2_input == NULL) {
      progress_finish_with_error(
          NETWORK_PROGRESS_STATUS_NULL_POINTER,
          "PTR_ERROR",
          layer_id,
          l2_input,
          l2_input_size,
          l2_output,
          l2_output_size,
          l2_weights,
          l2_weights_size,
          NULL,
          NULL,
          0,
          NULL,
          0,
          "L2_input is NULL");
      return -1;
    }

    if (!ptr_region_fits(l2_input, l2_input_size, l2_buffer, l2_buffer_size)) {
      progress_finish_with_error(
          NETWORK_PROGRESS_STATUS_POINTER_RANGE,
          "PTR_ERROR",
          layer_id,
          l2_input,
          l2_input_size,
          l2_output,
          l2_output_size,
          l2_weights,
          l2_weights_size,
          NULL,
          NULL,
          0,
          NULL,
          0,
          "L2_input outside scratch buffer");
      return -1;
    }
  }

  if (l2_weights_size > 0) {
    if (l2_weights == NULL) {
      progress_finish_with_error(
          NETWORK_PROGRESS_STATUS_NULL_POINTER,
          "PTR_ERROR",
          layer_id,
          l2_input,
          l2_input_size,
          l2_output,
          l2_output_size,
          l2_weights,
          l2_weights_size,
          NULL,
          NULL,
          0,
          NULL,
          0,
          "L2_weights is NULL");
      return -1;
    }

    if (!ptr_region_fits(l2_weights, l2_weights_size, l2_buffer, l2_buffer_size)) {
      progress_finish_with_error(
          NETWORK_PROGRESS_STATUS_POINTER_RANGE,
          "PTR_ERROR",
          layer_id,
          l2_input,
          l2_input_size,
          l2_output,
          l2_output_size,
          l2_weights,
          l2_weights_size,
          NULL,
          NULL,
          0,
          NULL,
          0,
          "L2_weights outside scratch buffer");
      return -1;
    }
  }

  return 0;
}

static int capture_final_tensor_chunk(
    int layer_id,
    const void *src,
    void *bbox_output,
    void *cls_output,
    size_t *bbox_offset,
    size_t *cls_offset) {
  if (layer_id >= NETWORK_BBOX_FIRST_LAYER && layer_id <= NETWORK_BBOX_LAST_LAYER && bbox_output != NULL) {
    if ((*bbox_offset + activations_out_size[layer_id]) > NETWORK_BBOX_OUTPUT_SIZE_BYTES) {
      return -1;
    }
    cl_ram_write((uint8_t *) bbox_output + *bbox_offset, (void *) src, activations_out_size[layer_id]);
    *bbox_offset += activations_out_size[layer_id];
  } else if (layer_id >= NETWORK_BBOX_FIRST_LAYER && layer_id <= NETWORK_BBOX_LAST_LAYER && bbox_output == NULL) {
    return -2;
  } else if (layer_id >= NETWORK_CLS_FIRST_LAYER && layer_id <= NETWORK_CLS_LAST_LAYER && cls_output != NULL) {
    if ((*cls_offset + activations_out_size[layer_id]) > NETWORK_CLS_OUTPUT_SIZE_BYTES) {
      return -1;
    }
    cl_ram_write((uint8_t *) cls_output + *cls_offset, (void *) src, activations_out_size[layer_id]);
    *cls_offset += activations_out_size[layer_id];
  } else if (layer_id >= NETWORK_CLS_FIRST_LAYER && layer_id <= NETWORK_CLS_LAST_LAYER && cls_output == NULL) {
    return -2;
  }

  return 0;
}
/* Moves the weights and the biases from hyperflash to hyperram */
void network_initialize() {

  L3_weights = ram_malloc(L3_WEIGHTS_SIZE);
  L3_input = ram_malloc(L3_INPUT_SIZE);
  L3_output = ram_malloc(L3_OUTPUT_SIZE);

#if NET_DEBUG_STARTUP
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_weights, L3_weights?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_input, L3_input?"Ok":"Failed");
  printf("\nL3 Buffer alloc initial\t@ %d:\t%s\n", (unsigned int)L3_output, L3_output?"Ok":"Failed");
#endif

  void *w_ptr = L3_weights;
  for (int i = 0; i < 60; i++) {
    size_t size = load_file_to_ram(w_ptr, L3_weights_files[i]);
    L3_weights_size[i] = size;
    w_ptr += size;
  }
}

/* Remove RAM memory */
void network_terminate() {
  ram_free(L3_weights, L3_WEIGHTS_SIZE);
  ram_free(L3_input, L3_INPUT_SIZE);
  ram_free(L3_output, L3_OUTPUT_SIZE);
}

void execute_layer_fork(void *args) {
  layer_args_t *layer_args = (layer_args_t *)args;
  if (pi_core_id() == 0) layer_args->L1_buffer = pmsis_l1_malloc(36700);

  switch (layer_args->layer_id)
  {
    case 0:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution0, args);
      break;
    case 1:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution1, args);
      break;
    case 2:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution2, args);
      break;
    case 3:
      pi_cl_team_fork(NUM_CORES, (void *)Addition3, args);
      break;
    case 4:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution4, args);
      break;
    case 5:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution5, args);
      break;
    case 6:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution6, args);
      break;
    case 7:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution7, args);
      break;
    case 8:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution8, args);
      break;
    case 9:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution9, args);
      break;
    case 10:
      pi_cl_team_fork(NUM_CORES, (void *)Addition10, args);
      break;
    case 11:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution11, args);
      break;
    case 12:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution12, args);
      break;
    case 13:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution13, args);
      break;
    case 14:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution14, args);
      break;
    case 15:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution15, args);
      break;
    case 16:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution16, args);
      break;
    case 17:
      pi_cl_team_fork(NUM_CORES, (void *)Addition17, args);
      break;
    case 18:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution18, args);
      break;
    case 19:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution19, args);
      break;
    case 20:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution20, args);
      break;
    case 21:
      pi_cl_team_fork(NUM_CORES, (void *)Addition21, args);
      break;
    case 22:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution22, args);
      break;
    case 23:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution23, args);
      break;
    case 24:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution24, args);
      break;
    case 25:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution25, args);
      break;
    case 26:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution26, args);
      break;
    case 27:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution27, args);
      break;
    case 28:
      pi_cl_team_fork(NUM_CORES, (void *)Addition28, args);
      break;
    case 29:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution29, args);
      break;
    case 30:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution30, args);
      break;
    case 31:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution31, args);
      break;
    case 32:
      pi_cl_team_fork(NUM_CORES, (void *)Addition32, args);
      break;
    case 33:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution33, args);
      break;
    case 34:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution34, args);
      break;
    case 35:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution35, args);
      break;
    case 36:
      pi_cl_team_fork(NUM_CORES, (void *)Addition36, args);
      break;
    case 37:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution37, args);
      break;
    case 38:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution38, args);
      break;
    case 39:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution39, args);
      break;
    case 40:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution40, args);
      break;
    case 41:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution41, args);
      break;
    case 42:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution42, args);
      break;
    case 43:
      pi_cl_team_fork(NUM_CORES, (void *)Addition43, args);
      break;
    case 44:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution44, args);
      break;
    case 45:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution45, args);
      break;
    case 46:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution46, args);
      break;
    case 47:
      pi_cl_team_fork(NUM_CORES, (void *)Addition47, args);
      break;
    case 48:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution48, args);
      break;
    case 49:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution49, args);
      break;
    case 50:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution50, args);
      break;
    case 51:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution51, args);
      break;
    case 52:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution52, args);
      break;
    case 53:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution53, args);
      break;
    case 54:
      pi_cl_team_fork(NUM_CORES, (void *)Addition54, args);
      break;
    case 55:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution55, args);
      break;
    case 56:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution56, args);
      break;
    case 57:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution57, args);
      break;
    case 58:
      pi_cl_team_fork(NUM_CORES, (void *)Addition58, args);
      break;
    case 59:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution59, args);
      break;
    case 60:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution60, args);
      break;
    case 61:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution61, args);
      break;
    case 62:
      pi_cl_team_fork(NUM_CORES, (void *)ReluConvolution62, args);
      break;
    case 63:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution63, args);
      break;
    case 64:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution64, args);
      break;
    case 65:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution65, args);
      break;
    case 66:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution66, args);
      break;
    case 67:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution67, args);
      break;
    case 68:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution68, args);
      break;
    case 69:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution69, args);
      break;
    case 70:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution70, args);
      break;
  }

  if (pi_core_id() == 0) pmsis_l1_malloc_free(layer_args->L1_buffer, 36700);
}

struct network_run_token network_run_outputs_async(
    void *l2_buffer,
    size_t l2_buffer_size,
    void *l2_final_output,
    struct network_final_outputs final_outputs,
    int exec,
    int initial_dir)
{
  struct pi_cluster_conf conf;

  if (g_network_async_state.active) {
    printf("ERROR: Network already running asynchronously.\n");
    pmsis_exit(-7);
  }

  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id = 0;
  g_network_async_state.args = (network_run_cluster_args_t) {
    .l2_buffer = (unsigned int) l2_buffer,
    .l2_buffer_size = (unsigned int) l2_buffer_size,
    .l2_final_output = (unsigned int) l2_final_output,
    .bbox_output = (unsigned int) final_outputs.bbox_output,
    .cls_output = (unsigned int) final_outputs.cls_output,
    .exec = (unsigned int) exec,
    .initial_dir = (unsigned int) initial_dir,
  };

  memset((void *) &g_network_async_state.cluster_dev, 0, sizeof(g_network_async_state.cluster_dev));
  memset((void *) &g_network_async_state.cluster_task, 0, sizeof(g_network_async_state.cluster_task));
  progress_reset();

  // open cluster...
  pi_cluster_task(&g_network_async_state.cluster_task, network_run_cluster, &g_network_async_state.args);
  pi_open_from_conf(&g_network_async_state.cluster_dev, &conf);
  if (pi_cluster_open(&g_network_async_state.cluster_dev)) {
    printf("ERROR: Cannot open cluster! Exiting...\n");
    pmsis_exit(-5);
  }
  // Then offload an entry point, this will get executed on the cluster controller
  g_network_async_state.cluster_task.stack_size = 3500;
  g_network_async_state.cluster_task.slave_stack_size = 3400;
  g_network_async_state.active = 1;
  pi_cluster_send_task_to_cl_async(
      &g_network_async_state.cluster_dev,
      &g_network_async_state.cluster_task,
      pi_task_block(&g_network_async_state.done_task));
  return (struct network_run_token) {
    .cluster_dev = g_network_async_state.cluster_dev
  };
}

struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)
{
  const struct network_final_outputs final_outputs = {
    .bbox_output = NULL,
    .cls_output = NULL,
  };
  return network_run_outputs_async(l2_buffer, l2_buffer_size, l2_final_output, final_outputs, exec, initial_dir);
}

void network_run_wait(struct network_run_token token)
{
  if (g_network_async_state.active) {
    pi_task_wait_on(&g_network_async_state.done_task);
    g_network_async_state.active = 0;
  }
  pi_cluster_close(&token.cluster_dev);
  print_perf("Final", cycle_network_execution, 12858368);
}

void network_run_outputs(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, struct network_final_outputs final_outputs, int exec, int initial_dir)
{
  network_run_wait(network_run_outputs_async(l2_buffer, l2_buffer_size, l2_final_output, final_outputs, exec, initial_dir));
}

void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)
{
  const struct network_final_outputs final_outputs = {
    .bbox_output = NULL,
    .cls_output = NULL,
  };
  network_run_outputs(l2_buffer, l2_buffer_size, l2_final_output, final_outputs, exec, initial_dir);
}

int network_get_progress_snapshot(struct network_progress_state *out) {
  uint32_t sequence_start = 0;
  uint32_t sequence_end = 0;

  if (out == NULL) {
    return -1;
  }

  do {
    sequence_start = g_network_progress.sequence;
    if (sequence_start & 1U) {
      continue;
    }

    out->sequence = sequence_start;
    out->started = g_network_progress.started;
    out->finished = g_network_progress.finished;
    out->latest_completed_layer = g_network_progress.latest_completed_layer;
    out->current_direction = g_network_progress.current_direction;
    out->heartbeat_count = g_network_progress.heartbeat_count;
    out->latest_output_size = g_network_progress.latest_output_size;
    out->latest_output_sample_size = g_network_progress.latest_output_sample_size;
    out->latest_output_sample_sum = g_network_progress.latest_output_sample_sum;
    out->cycle_count = g_network_progress.cycle_count;
    out->bbox_bytes_written = g_network_progress.bbox_bytes_written;
    out->cls_bytes_written = g_network_progress.cls_bytes_written;
    out->final_status = g_network_progress.final_status;
    out->marker_epoch = g_network_progress.marker_epoch;
    out->marker_layer = g_network_progress.marker_layer;
    out->l2_buffer_base_addr = g_network_progress.l2_buffer_base_addr;
    out->l2_buffer_total_size = g_network_progress.l2_buffer_total_size;
    memcpy(&out->allocator, (const void *) &g_network_progress.allocator, sizeof(out->allocator));
    memcpy(&out->layer66_output_alloc, (const void *) &g_network_progress.layer66_output_alloc, sizeof(out->layer66_output_alloc));
    memcpy(&out->layer67_output_alloc, (const void *) &g_network_progress.layer67_output_alloc, sizeof(out->layer67_output_alloc));
    out->allocator_begin_addr = g_network_progress.allocator_begin_addr;
    out->allocator_end_addr = g_network_progress.allocator_end_addr;
    out->l2_input_addr = g_network_progress.l2_input_addr;
    out->l2_input_size = g_network_progress.l2_input_size;
    out->l2_output_addr = g_network_progress.l2_output_addr;
    out->l2_output_size = g_network_progress.l2_output_size;
    out->l2_weights_addr = g_network_progress.l2_weights_addr;
    out->l2_weights_size = g_network_progress.l2_weights_size;
    out->l3_copy_src_addr = g_network_progress.l3_copy_src_addr;
    out->l3_copy_dst_addr = g_network_progress.l3_copy_dst_addr;
    out->l3_copy_size = g_network_progress.l3_copy_size;
    out->bbox_output_addr = g_network_progress.bbox_output_addr;
    out->cls_output_addr = g_network_progress.cls_output_addr;
    out->capture_dest_addr = g_network_progress.capture_dest_addr;
    out->capture_size = g_network_progress.capture_size;
    memcpy(out->latest_marker, (const void *) g_network_progress.latest_marker, sizeof(out->latest_marker));
    memcpy(out->latest_error, (const void *) g_network_progress.latest_error, sizeof(out->latest_error));

    sequence_end = g_network_progress.sequence;
  } while ((sequence_start != sequence_end) || (sequence_end & 1U));

  out->sequence = sequence_end;
  return 0;
}

const char *network_get_layer_name(int layer_id) {
  if (layer_id < 0 || layer_id >= NETWORK_NUM_LAYERS) {
    return "n/a";
  }

  return Layers_name[layer_id];
}

void network_run_cluster(void *args) {
  network_run_cluster_args_t *real_args = (network_run_cluster_args_t *) args;
  void * l2_buffer = (void *) real_args->l2_buffer;
  size_t l2_buffer_size = (size_t) real_args->l2_buffer_size;
  void * l2_final_output = (void *) real_args->l2_final_output;
  void * bbox_output = (void *) real_args->bbox_output;
  void * cls_output = (void *) real_args->cls_output;
  int exec = (int) real_args->exec;
  int dir = (int) real_args->initial_dir;
#if NET_DEBUG_STARTUP
  printf("network_run_cluster: l2_buffer=0x%08x size=%u exec=%d initial_dir=%d\n",
         (unsigned int) l2_buffer, (unsigned int) l2_buffer_size, exec, dir);
#endif
/*
  - initial buffer allocation L2 and L1
  - variable declaration
*/
/* ---------------------------------- */
/* -------- SECTION 0 BEGIN --------- */
/* ---------------------------------- */
  void *L2_output = NULL;
  void *L2_input = NULL;
  void *L2_weights = NULL;
  void *L3_weights_curr = L3_weights;
  void *bypass_activations = NULL;
  void *head_feature_caches[NETWORK_HEAD_SOURCE_COUNT] = {0};
  size_t head_feature_cache_sizes[NETWORK_HEAD_SOURCE_COUNT] = {0};

  int residual_number = 0;
  int bypass_dimension = 0;
  int perf_cyc = 0;
  size_t bbox_offset = 0;
  size_t cls_offset = 0;
  size_t current_input_alloc_size = 0;
  uint32_t final_status = NETWORK_PROGRESS_STATUS_SUCCESS;
/* ---------------------------------- */
/* --------- SECTION 0 END ---------- */
/* ---------------------------------- */

/*
  - initial copies from L3 of input
  - copies of weights of first 2 layers
*/
/* ---------------------------------- */
/* -------- SECTION 1 BEGIN --------- */
/* ---------------------------------- */
  directional_allocator_init(l2_buffer, l2_buffer_size);
  g_network_progress_ctx.l2_buffer_base_addr = (uint32_t) l2_buffer;
  g_network_progress_ctx.l2_buffer_total_size = (uint32_t) l2_buffer_size;
  g_network_progress_ctx.current_direction = (uint32_t) dir;
  g_network_progress_ctx.bbox_output_addr = (uint32_t) bbox_output;
  g_network_progress_ctx.cls_output_addr = (uint32_t) cls_output;
  g_network_progress_ctx.bbox_offset = 0;
  g_network_progress_ctx.cls_offset = 0;
  network_note_runtime_marker(
      "CLUSTER_READY",
      -1,
      NULL,
      0,
      NULL,
      0,
      NULL,
      0,
      NULL,
      NULL,
      0,
      NULL,
      0,
      NULL);
  for (int cache_idx = 0; cache_idx < NETWORK_HEAD_SOURCE_COUNT; cache_idx++) {
    const int source_layer = g_network_head_source_layers[cache_idx];
    head_feature_cache_sizes[cache_idx] = (size_t) activations_out_size[source_layer];
    network_note_runtime_marker(
        "CACHE_ALLOC_BEGIN",
        source_layer,
        NULL,
        0,
        NULL,
        0,
        NULL,
        0,
        NULL,
        NULL,
        head_feature_cache_sizes[cache_idx],
        NULL,
        0,
        NULL);
    head_feature_caches[cache_idx] = cl_ram_malloc(head_feature_cache_sizes[cache_idx]);
    if (head_feature_caches[cache_idx] == NULL) {
      final_status = NETWORK_PROGRESS_STATUS_HEAD_CACHE_ERROR;
      progress_finish_with_error(
          final_status,
          "CACHE_ERROR",
          source_layer,
          NULL,
          0,
          NULL,
          0,
          NULL,
          0,
          NULL,
          NULL,
          head_feature_cache_sizes[cache_idx],
          NULL,
          0,
          "Head feature cache allocation failed");
      goto network_run_cluster_abort;
    }
    network_note_runtime_marker(
        "CACHE_ALLOC_OK",
        source_layer,
        NULL,
        0,
        NULL,
        0,
        NULL,
        0,
        NULL,
        head_feature_caches[cache_idx],
        head_feature_cache_sizes[cache_idx],
        NULL,
        0,
        NULL);
  }
  network_note_runtime_marker(
      "POST_CACHE_SETUP",
      -1,
      NULL,
      0,
      NULL,
      0,
      NULL,
      0,
      NULL,
      NULL,
      0,
      NULL,
      0,
      NULL);

/* ---------------------------------- */
/* --------- SECTION 1 END ---------- */
/* ---------------------------------- */
  // perf measurement begin
  cycle_network_execution = 0;
/* MAIN SECTION
  - for loop over all the layers of the network
  - double buffering using L3
  - check on layers to be executed from L3
  - residual check at the end of each layer
*/
/* ---------------------------------- */
/* -------- SECTION 2 BEGIN --------- */
/* ---------------------------------- */
  int weight_l_cnt = 0; // count how many layers with weights we have processed to increment the weights_L3 pointer
  network_note_runtime_marker(
      "PRE_LAYER_LOOP",
      -1,
      NULL,
      0,
      NULL,
      0,
      NULL,
      0,
      NULL,
      NULL,
      0,
      NULL,
      0,
      NULL);
  for (int i = 0; i < NETWORK_NUM_LAYERS; i++) {
    void *capture_dest = NULL;
    size_t capture_size = 0;
    size_t layer_input_size = current_input_alloc_size;
    int head_cache_index = network_head_cache_index_for_layer(i);
/* MEMORY ALLOCATION
  - allocate memory if layer is executed from L3;
  - allocate weights
  - read weights
*/
    if (i == 0 || i == 67) {
      network_note_runtime_marker(
          i == 0 ? "LAYER0_PREP_BEGIN" : "L67_PRE_SETUP",
          i,
          L2_input,
          layer_input_size,
          L2_output,
          0,
          L2_weights,
          0,
          NULL,
          NULL,
          0,
          NULL,
          0,
          NULL);
    }

    L2_output = dmalloc(activations_out_size[i], !dir);
    progress_record_output_alloc_probe(i);
    if (i == 0 || i == 66 || i == 67) {
      network_note_runtime_marker(
          i == 0 ? "LAYER0_OUTPUT_READY" : (i == 66 ? "L66_OUTPUT_ALLOC" : "L67_OUTPUT_ALLOC"),
          i,
          L2_input,
          layer_input_size,
          L2_output,
          activations_out_size[i],
          NULL,
          0,
          NULL,
          NULL,
          0,
          NULL,
          0,
          NULL);
    }
    if (L2_output == NULL) {
      final_status = NETWORK_PROGRESS_STATUS_ALLOC_FAILURE;
      progress_finish_with_error(
          final_status,
          "ALLOC_ERROR",
          i,
          L2_input,
          layer_input_size,
          L2_output,
          activations_out_size[i],
          NULL,
          0,
          NULL,
          NULL,
          0,
          NULL,
          0,
          "L2_output allocation returned NULL");
      goto network_run_cluster_abort;
    }

    if (L3_input_layers[i] == 1) {
      L2_input = dmalloc(activations_size[i], dir);
      current_input_alloc_size = activations_size[i];
      if (i == 0) {
        network_note_runtime_marker(
            "LAYER0_INPUT_READY",
            i,
            L2_input,
            current_input_alloc_size,
            L2_output,
            activations_out_size[i],
            L2_weights,
            0,
            NULL,
            NULL,
            0,
            NULL,
            0,
            NULL);
      }
    } else if (head_cache_index >= 0) {
      if (head_feature_caches[head_cache_index] == NULL) {
        final_status = NETWORK_PROGRESS_STATUS_HEAD_CACHE_ERROR;
        progress_finish_with_error(
            final_status,
            "CACHE_ERROR",
            i,
            NULL,
            0,
            L2_output,
            activations_out_size[i],
            NULL,
            0,
            NULL,
            NULL,
            0,
            NULL,
            0,
            "Head feature cache pointer is NULL");
        goto network_run_cluster_abort;
      }
      L2_input = dmalloc(activations_size[i], dir);
      current_input_alloc_size = activations_size[i];
      if (L2_input == NULL) {
        final_status = NETWORK_PROGRESS_STATUS_ALLOC_FAILURE;
        progress_finish_with_error(
            final_status,
            "ALLOC_ERROR",
            i,
            L2_input,
            current_input_alloc_size,
            L2_output,
            activations_out_size[i],
            NULL,
            0,
            NULL,
            NULL,
            0,
            NULL,
            0,
            "L2_input allocation returned NULL");
        goto network_run_cluster_abort;
      }
      cl_ram_read(L2_input, head_feature_caches[head_cache_index], activations_size[i]);
    }
    layer_input_size = current_input_alloc_size;

    if (layer_with_weights[i] == 1) {
      L2_weights = dmalloc(weights_size[i], dir);
      if (L2_weights == NULL) {
        final_status = NETWORK_PROGRESS_STATUS_ALLOC_FAILURE;
        progress_finish_with_error(
            final_status,
            "ALLOC_ERROR",
            i,
            L2_input,
            layer_input_size,
            L2_output,
            activations_out_size[i],
            L2_weights,
            weights_size[i],
            NULL,
            NULL,
            0,
            NULL,
            0,
            "L2_weights allocation returned NULL");
        goto network_run_cluster_abort;
      }
      if (i == 0) {
        network_note_runtime_marker(
            "LAYER0_WEIGHTS_READY",
            i,
            L2_input,
            layer_input_size,
            L2_output,
            activations_out_size[i],
            L2_weights,
            weights_size[i],
            NULL,
            NULL,
            0,
            NULL,
            0,
            NULL);
      }
    }

    if (i == 0 || i == 67) {
      network_note_runtime_marker(
          i == 0 ? "LAYER0_PTRS_READY" : "L67_PTRS_READY",
          i,
          L2_input,
          layer_input_size,
          L2_output,
          activations_out_size[i],
          L2_weights,
          layer_with_weights[i] == 1 ? weights_size[i] : 0,
          NULL,
          NULL,
          0,
          NULL,
          0,
          NULL);
    }

    if (validate_layer_l2_pointers(
            i,
            l2_buffer,
            l2_buffer_size,
            L2_input,
            layer_input_size,
            L2_output,
            activations_out_size[i],
            L2_weights,
            layer_with_weights[i] == 1 ? weights_size[i] : 0) != 0) {
      final_status = g_network_progress.final_status;
      goto network_run_cluster_abort;
    }

    if (allocate_layer[i] == 1) {
      if (i == 0 || i == 67) {
        network_note_runtime_marker(
            i == 0 ? "L0_BEFORE_WEIGHT_LOAD" : "L67_BEFORE_WEIGHT_LOAD",
            i,
            L2_input,
            activations_size[i],
            L2_output,
            activations_out_size[i],
            L2_weights,
            weights_size[i],
            L3_weights_curr,
            L2_weights,
            weights_size[i],
            NULL,
            0,
            NULL);
      }
      cl_ram_read(L2_weights, L3_weights_curr, weights_size[i]);
      if (i == 0 || i == 67) {
        network_note_runtime_marker(
            i == 0 ? "L0_AFTER_WEIGHT_LOAD" : "L67_AFTER_WEIGHT_LOAD",
            i,
            L2_input,
            activations_size[i],
            L2_output,
            activations_out_size[i],
            L2_weights,
            weights_size[i],
            L3_weights_curr,
            L2_weights,
            weights_size[i],
            NULL,
            0,
            NULL);
      }
    }

#if NET_DEBUG_LAYER0_SUMMARY
    if (i == 0) {
      print_u8_summary("Layer0 L2_input", (const uint8_t *) L2_input, activations_size[i], 32);
    }
#endif

#if NET_DEBUG_LAYER_CHECKSUMS
    if (L3_input_layers[i] == 1)
      printf("Input in L3\n");
    else
    if (i == 0 || branch_change[i-1] == 0) {
      checksum("L2 input", L2_input, activations_size[i], activations_checksum[i][exec]);
      if (allocate_layer[i] == 1)
        checksum("L2 weights", L2_weights, weights_size[i], weights_checksum[i]);
      else
        printf("Weights in L3\n");
    }
    else
      printf("Switching branch, already checked activation\n");
#endif

    layer_args_t largs = {
      .L3_input = (unsigned int) L3_input,
      .L3_output = (unsigned int) L3_output,
      .L3_after_weights = (unsigned int) L3_weights_curr,
      .L2_input = (unsigned int) L2_input,
      .bypass = (unsigned int) bypass_activations,
      .L2_output = (unsigned int) L2_output,
      .L2_weights = (unsigned int) L2_weights,
      .L1_buffer = 0,
      .ram = (unsigned int) get_ram_ptr(),
      .out_mult = (unsigned int) out_mult_vector[i],
      .out_shift = (unsigned int) out_shift_vector[i],
      .layer_id = i
    };

/*
- Execution of the layers_pointers
*/
    if (i == 0 || i == 67) {
      network_note_runtime_marker(
          i == 0 ? "L0_BEFORE_EXECUTE" : "L67_BEFORE_EXECUTE",
          i,
          L2_input,
          activations_size[i],
          L2_output,
          activations_out_size[i],
          L2_weights,
          layer_with_weights[i] == 1 ? weights_size[i] : 0,
          L3_weights_curr,
          L2_weights,
          allocate_layer[i] == 1 ? weights_size[i] : 0,
          NULL,
          0,
          NULL);
    }

    // perf measurement begin
    pi_perf_conf(1<<PI_PERF_CYCLES);
    pi_perf_reset();
    pi_perf_stop();
    pi_perf_start();
    execute_layer_fork((void *) &largs);
    // performance measurements: end
    pi_perf_stop();
    perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
    cycle_network_execution += perf_cyc;

    if (i == 0 || i == 67) {
      network_note_runtime_marker(
          i == 0 ? "L0_AFTER_EXECUTE" : "L67_AFTER_EXECUTE",
          i,
          L2_input,
          activations_size[i],
          L2_output,
          activations_out_size[i],
          L2_weights,
          layer_with_weights[i] == 1 ? weights_size[i] : 0,
          NULL,
          NULL,
          0,
          NULL,
          0,
          NULL);
    }

    asm volatile("": : :"memory");
    unsigned int temp = L3_input;
    L3_input = L3_output;
    asm volatile("": : :"memory");
    L3_output = temp;
    asm volatile("": : :"memory");

#if NET_DEBUG_LAYER0_SUMMARY
    if (i == 0) {
      print_u8_summary("Layer0 output", (const uint8_t *) L2_output, activations_out_size[i], 32);
    }
#endif

    {
      int source_cache_index = network_head_cache_index_from_source_layer(i);
      if (source_cache_index >= 0) {
        if (head_feature_caches[source_cache_index] == NULL) {
          final_status = NETWORK_PROGRESS_STATUS_HEAD_CACHE_ERROR;
          progress_finish_with_error(
              final_status,
              "CACHE_ERROR",
              i,
              L2_input,
              layer_input_size,
              L2_output,
              activations_out_size[i],
              L2_weights,
              layer_with_weights[i] == 1 ? weights_size[i] : 0,
              NULL,
              NULL,
              activations_out_size[i],
              NULL,
              0,
              "Head feature cache pointer is NULL during write");
          goto network_run_cluster_abort;
        }
        cl_ram_write(head_feature_caches[source_cache_index], L2_output, activations_out_size[i]);
      }
    }

    if (i >= NETWORK_BBOX_FIRST_LAYER && i <= NETWORK_BBOX_LAST_LAYER && bbox_output != NULL) {
      capture_dest = (uint8_t *) bbox_output + bbox_offset;
      capture_size = activations_out_size[i];
    } else if (i >= NETWORK_CLS_FIRST_LAYER && i <= NETWORK_CLS_LAST_LAYER && cls_output != NULL) {
      capture_dest = (uint8_t *) cls_output + cls_offset;
      capture_size = activations_out_size[i];
    }

    {
      int capture_status = capture_final_tensor_chunk(i, L2_output, bbox_output, cls_output, &bbox_offset, &cls_offset);
      if (capture_status != 0) {
        final_status = capture_status == -1 ? NETWORK_PROGRESS_STATUS_CAPTURE_OVERFLOW : NETWORK_PROGRESS_STATUS_CAPTURE_NULL_DEST;
        progress_finish_with_error(
            final_status,
            "CAPTURE_ERROR",
            i,
            L2_input,
            activations_size[i],
            L2_output,
            activations_out_size[i],
            L2_weights,
            layer_with_weights[i] == 1 ? weights_size[i] : 0,
            NULL,
            NULL,
            0,
            capture_dest,
            capture_size,
            capture_status == -1 ? "Final tensor capture exceeds reserved bytes" : "Final tensor capture destination is NULL");
        goto network_run_cluster_abort;
      }
    }

    g_network_progress_ctx.bbox_offset = (uint32_t) bbox_offset;
    g_network_progress_ctx.cls_offset = (uint32_t) cls_offset;

    if (i == 66) {
      network_note_runtime_marker(
          "L66_COMPLETE",
          i,
          L2_input,
          activations_size[i],
          L2_output,
          activations_out_size[i],
          L2_weights,
          layer_with_weights[i] == 1 ? weights_size[i] : 0,
          NULL,
          NULL,
          0,
          capture_dest,
          capture_size,
          NULL);
    }

    if (i == 67) {
      network_note_runtime_marker(
          "L67_AFTER_CAPTURE",
          i,
          L2_input,
          activations_size[i],
          L2_output,
          activations_out_size[i],
          L2_weights,
          layer_with_weights[i] == 1 ? weights_size[i] : 0,
          NULL,
          NULL,
          0,
          capture_dest,
          capture_size,
          NULL);
    }

    progress_publish(
        i,
        (const uint8_t *) L2_output,
        activations_out_size[i],
        (uint32_t) cycle_network_execution,
        bbox_offset,
        cls_offset,
        ((i == 0) || (((i + 1) % NET_PROGRESS_HEARTBEAT_INTERVAL) == 0) || (i >= NETWORK_BBOX_FIRST_LAYER)),
        (final_status == NETWORK_PROGRESS_STATUS_CAPTURE_OVERFLOW) ? final_status : NETWORK_PROGRESS_STATUS_RUNNING,
        0);

#if NET_DEBUG_PROGRESS_INTERVAL > 0
    if (((i + 1) % NET_DEBUG_PROGRESS_INTERVAL) == 0 || i >= NETWORK_BBOX_FIRST_LAYER) {
      printf("Completed layer %d (%s)\n", i, Layers_name[i]);
    }
#endif

#if NET_DEBUG_LAYER_CHECKSUMS
    printf("Layer %s %d ended: \n", Layers_name[i], i);
    if (L3_output_layers[i]==1) {
      printf("Output in L3. Expected checksum: %d\n", activations_out_checksum[i][exec]);
    } else {
      checksum(i + 1 < 71 ? "L2 output" : "final output",
               L2_output, activations_out_size[i], activations_out_checksum[i][exec]);
    }
    printf("\n");
#endif

    // Free memory
    if (layer_with_weights[i] == 1) {
      dfree(weights_size[i], dir);
      L2_weights = NULL;
    }
    if (current_input_alloc_size > 0) {
      dfree((int) current_input_alloc_size, dir);
      current_input_alloc_size = 0;
    }
    if (branch_input[i] == 1)
      dfree(bypass_dimension, dir);
    L2_input = NULL;
    // Residual connections
    if (i < 70) {
      int carry_output_to_next_input = !network_layer_uses_head_cache_input(i + 1);
      int output_freed = 0;

      if (branch_input[i+1] == 1) {
        bypass_activations = dmalloc(bypass_dimension, !dir);
        if (bypass_activations == NULL) {
          final_status = NETWORK_PROGRESS_STATUS_ALLOC_FAILURE;
          progress_finish_with_error(
              final_status,
              "ALLOC_ERROR",
              i,
              L2_input,
              current_input_alloc_size,
              L2_output,
              activations_out_size[i],
              L2_weights,
              0,
              NULL,
              NULL,
              bypass_dimension,
              NULL,
              0,
              "Bypass allocation returned NULL");
          goto network_run_cluster_abort;
        }
        residual_number--;
        cl_ram_read(bypass_activations, layers_pointers[residual_number], bypass_dimension);
        cl_ram_free(layers_pointers[residual_number], bypass_dimension);
      }

      // TODO I feel like this should look ahead instead of back
      if (i > 0 && branch_output[i-1]==1 && L3_input_layers[i]==1) { // TODO don't understand this condition
        L3_input = cl_ram_malloc(1500000);
      }
      if (branch_output[i]==1 && L3_output_layers[i]==1) {
        cl_ram_free(L3_input + activations_out_size[i], 1500000 - activations_out_size[i]);
        layers_pointers[residual_number] = L3_input;
        residual_number++;
        bypass_dimension = activations_out_size[i];
      } else
    if (branch_output[i]==1 || branch_change[i] == 1) {
        layers_pointers[residual_number] = cl_ram_malloc(activations_out_size[i]);
        cl_ram_write(layers_pointers[residual_number], L2_output, activations_out_size[i]);
        residual_number++;
        bypass_dimension = activations_out_size[i];
    }

      if (branch_change[i]==1) {
        dfree(activations_out_size[i], !dir);
        output_freed = 1;
        L2_output = NULL;
        L2_input = dmalloc(activations_size[i + 1], !dir);
        if (L2_input == NULL) {
          final_status = NETWORK_PROGRESS_STATUS_ALLOC_FAILURE;
          progress_finish_with_error(
              final_status,
              "ALLOC_ERROR",
              i,
              L2_input,
              activations_size[i + 1],
              NULL,
              activations_out_size[i],
              NULL,
              0,
              NULL,
              NULL,
              0,
              NULL,
              0,
              "Branch-change input allocation returned NULL");
          goto network_run_cluster_abort;
        }
        current_input_alloc_size = activations_size[i + 1];
        cl_ram_read(L2_input, layers_pointers[residual_number - 2], activations_size[i + 1]);
        cl_ram_free(layers_pointers[residual_number - 2], activations_size[i + 1]);
      } else if (L3_output_layers[i] == 1) {
        dfree(activations_out_size[i], !dir);
        output_freed = 1;
        L2_output = NULL;
      } else if (carry_output_to_next_input) {
        L2_input = L2_output;
        current_input_alloc_size = activations_out_size[i];
      }

      if (!output_freed && !carry_output_to_next_input) {
        dfree(activations_out_size[i], !dir);
        output_freed = 1;
        L2_output = NULL;
      }
    }
    if (layer_with_weights[i])
       L3_weights_curr += L3_weights_size[weight_l_cnt++];
    dir = !dir;
    g_network_progress_ctx.current_direction = (uint32_t) dir;
  }

network_run_cluster_abort:
  for (int cache_idx = 0; cache_idx < NETWORK_HEAD_SOURCE_COUNT; cache_idx++) {
    if (head_feature_caches[cache_idx] != NULL) {
      cl_ram_free(head_feature_caches[cache_idx], head_feature_cache_sizes[cache_idx]);
      head_feature_caches[cache_idx] = NULL;
    }
  }

  if (l2_final_output != NULL && L2_output != NULL) {
    for (int i = 0; i < activations_out_size[70]; i++) {
      ((uint8_t *) l2_final_output)[i] = ((uint8_t *) L2_output)[i];
    }
  }

  if (final_status == NETWORK_PROGRESS_STATUS_SUCCESS) {
    progress_publish(
        NETWORK_NUM_LAYERS - 1,
        (const uint8_t *) L2_output,
        activations_out_size[NETWORK_NUM_LAYERS - 1],
        (uint32_t) cycle_network_execution,
        bbox_offset,
        cls_offset,
        1,
        final_status,
        1);
  }

/* ---------------------------------- */
/* --------- SECTION 2 END ---------- */
/* ---------------------------------- */

/* ---------------------------------- */
/* -------- SECTION 3 BEGIN --------- */
/* ---------------------------------- */


/* ---------------------------------- */
/* --------- SECTION 3 END ---------- */
/* ---------------------------------- */
}
