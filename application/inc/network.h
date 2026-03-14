/*
 * network.h
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

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <stddef.h>
#include <stdint.h>
#include "pmsis.h"

#define NETWORK_NUM_LAYERS 71
#define NETWORK_BBOX_OUTPUT_SIZE_BYTES (131072 + 32768 + 8192 + 2048)
#define NETWORK_CLS_OUTPUT_SIZE_BYTES (65536 + 16384 + 4096 + 1024)
#define NETWORK_BBOX_OUTPUT_COUNT (NETWORK_BBOX_OUTPUT_SIZE_BYTES / sizeof(int32_t))
#define NETWORK_CLS_OUTPUT_COUNT (NETWORK_CLS_OUTPUT_SIZE_BYTES / sizeof(int32_t))
#define NETWORK_PROGRESS_MARKER_TEXT_SIZE 32
#define NETWORK_PROGRESS_ERROR_TEXT_SIZE 64

struct network_final_outputs {
  void *bbox_output;
  void *cls_output;
};

struct network_allocator_state {
  uint32_t base_addr;
  uint32_t limit_addr;
  uint32_t begin_addr;
  uint32_t end_addr;
  uint32_t last_before_begin_addr;
  uint32_t last_before_end_addr;
  uint32_t last_after_begin_addr;
  uint32_t last_after_end_addr;
  uint32_t last_return_addr;
  uint32_t last_request_size;
  uint32_t last_alignment;
  int32_t last_direction;
  int32_t last_operation;
};

struct network_output_alloc_probe {
  uint32_t valid;
  int32_t layer_id;
  struct network_allocator_state allocator;
};

struct network_run_token {
  struct pi_device cluster_dev;
};

struct network_progress_state {
  uint32_t sequence;
  uint32_t started;
  uint32_t finished;
  int32_t latest_completed_layer;
  uint32_t heartbeat_count;
  uint32_t latest_output_size;
  uint32_t latest_output_sample_size;
  uint32_t latest_output_sample_sum;
  uint32_t cycle_count;
  uint32_t bbox_bytes_written;
  uint32_t cls_bytes_written;
  uint32_t final_status;
  uint32_t marker_epoch;
  int32_t marker_layer;
  uint32_t l2_buffer_base_addr;
  uint32_t l2_buffer_total_size;
  struct network_allocator_state allocator;
  struct network_output_alloc_probe layer66_output_alloc;
  struct network_output_alloc_probe layer67_output_alloc;
  uint32_t allocator_begin_addr;
  uint32_t allocator_end_addr;
  uint32_t l2_input_addr;
  uint32_t l2_input_size;
  uint32_t l2_output_addr;
  uint32_t l2_output_size;
  uint32_t l2_weights_addr;
  uint32_t l2_weights_size;
  uint32_t l3_copy_src_addr;
  uint32_t l3_copy_dst_addr;
  uint32_t l3_copy_size;
  uint32_t bbox_output_addr;
  uint32_t cls_output_addr;
  uint32_t capture_dest_addr;
  uint32_t capture_size;
  char latest_marker[NETWORK_PROGRESS_MARKER_TEXT_SIZE];
  char latest_error[NETWORK_PROGRESS_ERROR_TEXT_SIZE];
};


void network_terminate();
void network_initialize();
void network_run_cluster(void * args);
struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
struct network_run_token network_run_outputs_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, struct network_final_outputs final_outputs, int exec, int initial_dir);
void network_run_wait(struct network_run_token token);
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void network_run_outputs(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, struct network_final_outputs final_outputs, int exec, int initial_dir);
void execute_layer_fork(void *arg);
int network_get_progress_snapshot(struct network_progress_state *out);
const char *network_get_layer_name(int layer_id);
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
    const char *error_text);


#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
static const char * L3_weights_files[] = {
  "ReluConvolution0_weights.hex", "ReluConvolution1_weights.hex", "Convolution2_weights.hex", "ReluConvolution4_weights.hex", "ReluConvolution5_weights.hex", "Convolution6_weights.hex", "ReluConvolution7_weights.hex", "ReluConvolution8_weights.hex", "Convolution9_weights.hex", "ReluConvolution11_weights.hex", "ReluConvolution12_weights.hex", "Convolution13_weights.hex", "ReluConvolution14_weights.hex", "ReluConvolution15_weights.hex", "Convolution16_weights.hex", "ReluConvolution18_weights.hex", "ReluConvolution19_weights.hex", "Convolution20_weights.hex", "ReluConvolution22_weights.hex", "ReluConvolution23_weights.hex", "Convolution24_weights.hex", "ReluConvolution25_weights.hex", "ReluConvolution26_weights.hex", "Convolution27_weights.hex", "ReluConvolution29_weights.hex", "ReluConvolution30_weights.hex", "Convolution31_weights.hex", "ReluConvolution33_weights.hex", "ReluConvolution34_weights.hex", "Convolution35_weights.hex", "ReluConvolution37_weights.hex", "ReluConvolution38_weights.hex", "Convolution39_weights.hex", "ReluConvolution40_weights.hex", "ReluConvolution41_weights.hex", "Convolution42_weights.hex", "ReluConvolution44_weights.hex", "ReluConvolution45_weights.hex", "Convolution46_weights.hex", "ReluConvolution48_weights.hex", "ReluConvolution49_weights.hex", "Convolution50_weights.hex", "ReluConvolution51_weights.hex", "ReluConvolution52_weights.hex", "Convolution53_weights.hex", "ReluConvolution55_weights.hex", "ReluConvolution56_weights.hex", "Convolution57_weights.hex", "ReluConvolution59_weights.hex", "ReluConvolution60_weights.hex", "Convolution61_weights.hex", "ReluConvolution62_weights.hex", "Convolution63_weights.hex", "Convolution64_weights.hex", "Convolution65_weights.hex", "Convolution66_weights.hex", "Convolution67_weights.hex", "Convolution68_weights.hex", "Convolution69_weights.hex", "Convolution70_weights.hex"
};
static int L3_weights_size[60];
static int layers_pointers[71];
static char * Layers_name[71] = {"ReluConvolution0", "ReluConvolution1", "Convolution2", "Addition3", "ReluConvolution4", "ReluConvolution5", "Convolution6", "ReluConvolution7", "ReluConvolution8", "Convolution9", "Addition10", "ReluConvolution11", "ReluConvolution12", "Convolution13", "ReluConvolution14", "ReluConvolution15", "Convolution16", "Addition17", "ReluConvolution18", "ReluConvolution19", "Convolution20", "Addition21", "ReluConvolution22", "ReluConvolution23", "Convolution24", "ReluConvolution25", "ReluConvolution26", "Convolution27", "Addition28", "ReluConvolution29", "ReluConvolution30", "Convolution31", "Addition32", "ReluConvolution33", "ReluConvolution34", "Convolution35", "Addition36", "ReluConvolution37", "ReluConvolution38", "Convolution39", "ReluConvolution40", "ReluConvolution41", "Convolution42", "Addition43", "ReluConvolution44", "ReluConvolution45", "Convolution46", "Addition47", "ReluConvolution48", "ReluConvolution49", "Convolution50", "ReluConvolution51", "ReluConvolution52", "Convolution53", "Addition54", "ReluConvolution55", "ReluConvolution56", "Convolution57", "Addition58", "ReluConvolution59", "ReluConvolution60", "Convolution61", "ReluConvolution62", "Convolution63", "Convolution64", "Convolution65", "Convolution66", "Convolution67", "Convolution68", "Convolution69", "Convolution70"};
static int L3_input_layers[71] = {1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[71] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[71] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static int branch_input[71] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[71] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[71] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[71] = {7522, 9672, 7591, 0, 52828, 49904, 46921, 51979, 56434, 46610, 0, 48574, 53496, 48664, 49960, 52421, 49670, 0, 49785, 55967, 49055, 0, 48298, 57409, 49276, 47279, 59030, 51521, 0, 46430, 59731, 48499, 0, 46109, 62498, 47112, 0, 46361, 57430, 95897, 192588, 112363, 194432, 0, 191242, 113895, 197012, 0, 196588, 118528, 196914, 195222, 115450, 196382, 0, 198881, 113100, 197091, 0, 196625, 115754, 386107, 524431, 295972, 276954, 561948, 4722122, 150592, 149100, 303755, 2304055};
static int weights_size[71] = {72, 72, 64, 0, 384, 432, 384, 384, 432, 384, 0, 384, 432, 384, 384, 432, 384, 0, 384, 432, 384, 0, 384, 432, 384, 384, 432, 384, 0, 384, 432, 384, 0, 384, 432, 384, 0, 384, 432, 768, 1536, 864, 1536, 0, 1536, 864, 1536, 0, 1536, 864, 1536, 1536, 864, 1536, 0, 1536, 864, 1536, 0, 1536, 864, 3072, 4096, 2432, 2432, 4736, 36992, 1216, 1216, 2368, 18496};
static int activations_checksum[71][1] = {{
  2090675  },
{
  4504471  },
{
  2273806  },
{
  254940  },
{
  4847475  },
{
  26317430  },
{
  4786584  },
{
  1059278  },
{
  6067215  },
{
  2700719  },
{
  -138909439  },
{
  1053135  },
{
  5724240  },
{
  1030004  },
{
  249021  },
{
  1575067  },
{
  947703  },
{
  20905744  },
{
  257997  },
{
  1709010  },
{
  1103373  },
{
  -20977549  },
{
  259136  },
{
  1750575  },
{
  298730  },
{
  66287  },
{
  361335  },
{
  260072  },
{
  -11216323  },
{
  66092  },
{
  349860  },
{
  333617  },
{
  2328644  },
{
  63344  },
{
  399075  },
{
  243425  },
{
  4777325  },
{
  64733  },
{
  341700  },
{
  335351  },
{
  131729  },
{
  780045  },
{
  572294  },
{
  19749107  },
{
  131204  },
{
  771375  },
{
  470145  },
{
  -9386308  },
{
  129344  },
{
  777495  },
{
  119438  },
{
  34074  },
{
  172380  },
{
  144195  },
{
  5081267  },
{
  34765  },
{
  186915  },
{
  162948  },
{
  -556381  },
{
  32880  },
{
  190230  },
{
  160137  },
{
  66196  },
{
  211395  },
{
  4167288  },
{
  1044828  },
{
  263591  },
{
  5543913  },
{
  2063180  },
{
  522336  },
{
  134099  }
};
static int activations_size[71] = {16384, 32768, 32768, 131072, 32768, 196608, 49152, 8192, 49152, 49152, 32768, 8192, 49152, 12288, 2048, 12288, 12288, 8192, 2048, 12288, 12288, 8192, 2048, 12288, 3072, 512, 3072, 3072, 2048, 512, 3072, 3072, 2048, 512, 3072, 3072, 2048, 512, 3072, 3072, 1024, 6144, 6144, 4096, 1024, 6144, 6144, 4096, 1024, 6144, 1536, 256, 1536, 1536, 1024, 256, 1536, 1536, 1024, 256, 1536, 1536, 512, 8192, 2048, 1024, 2048, 8192, 2048, 1024, 2048};
static int out_mult_vector[71] = {38, 49, 1, 1, 39, 56, 1, 43, 37, 1, 1, 37, 41, 1, 47, 35, 1, 1, 52, 38, 1, 1, 56, 45, 1, 57, 41, 1, 1, 32, 44, 1, 1, 60, 56, 1, 1, 32, 51, 1, 59, 42, 1, 1, 63, 45, 1, 1, 49, 51, 1, 57, 48, 1, 1, 56, 46, 1, 1, 57, 59, 1, 35, 1, 1, 1, 1, 1, 1, 1, 1};
static int out_shift_vector[71] = {5, 6, 0, 0, 6, 8, 0, 7, 8, 0, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 7, 0, 0, 6, 7, 0, 0, 7, 7, 0, 0, 6, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0};
static int activations_out_checksum[71][1] = {{
  4504471 },
{
  2273806 },
{
  15082415 },
{
  15097611 },
{
  26317430 },
{
  4786584 },
{
  3857670 },
{
  6067215 },
{
  2700719 },
{
  5366507 },
{
  4039617 },
{
  5724240 },
{
  1030004 },
{
  905952 },
{
  1575067 },
{
  947703 },
{
  969186 },
{
  852894 },
{
  1709010 },
{
  1103373 },
{
  1128569 },
{
  934416 },
{
  1750575 },
{
  298730 },
{
  285739 },
{
  361335 },
{
  260072 },
{
  308750 },
{
  321102 },
{
  349860 },
{
  333617 },
{
  252928 },
{
  300135 },
{
  399075 },
{
  243425 },
{
  240069 },
{
  288545 },
{
  341700 },
{
  335351 },
{
  524207 },
{
  780045 },
{
  572294 },
{
  451844 },
{
  467610 },
{
  771375 },
{
  470145 },
{
  562081 },
{
  503359 },
{
  777495 },
{
  119438 },
{
  125233 },
{
  172380 },
{
  144195 },
{
  116261 },
{
  122300 },
{
  186915 },
{
  162948 },
{
  130460 },
{
  124124 },
{
  190230 },
{
  160137 },
{
  254755 },
{
  211395 },
{
  18905806 },
{
  4607685 },
{
  1054644 },
{
  24797574 },
{
  9230394 },
{
  2121497 },
{
  490986 },
{
  11972288 }
};
static int activations_out_size[71] = {32768, 32768, 131072, 131072, 196608, 49152, 32768, 49152, 49152, 32768, 32768, 49152, 12288, 8192, 12288, 12288, 8192, 8192, 12288, 12288, 8192, 8192, 12288, 3072, 2048, 3072, 3072, 2048, 2048, 3072, 3072, 2048, 2048, 3072, 3072, 2048, 2048, 3072, 3072, 4096, 6144, 6144, 4096, 4096, 6144, 6144, 4096, 4096, 6144, 1536, 1024, 1536, 1536, 1024, 1024, 1536, 1536, 1024, 1024, 1536, 1536, 2048, 2048, 131072, 32768, 8192, 2048, 65536, 16384, 4096, 1024};
static int layer_with_weights[71] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#endif

#endif  // __NETWORK_H__
