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
#include "pmsis.h"


struct network_run_token {
  struct pi_device cluster_dev;
};


void network_terminate();
void network_initialize();
void network_run_cluster(void * args);
struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void network_run_wait(struct network_run_token token);
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir);
void execute_layer_fork(void *arg);


#ifdef DEFINE_CONSTANTS
// allocation of buffers with parameters needed by the network execution
static const char * L3_weights_files[] = {
  "ReluConvolution0_weights.hex", "ReluConvolution1_weights.hex", "Convolution2_weights.hex", "Convolution3_weights.hex", "ReluConvolution5_weights.hex", "ReluConvolution6_weights.hex", "Convolution7_weights.hex", "Convolution8_weights.hex", "ReluConvolution10_weights.hex", "ReluConvolution11_weights.hex", "ReluConvolution12_weights.hex", "FullyConnected14_weights.hex"
};
static int L3_weights_size[12];
static int layers_pointers[15];
static char * Layers_name[15] = {"ReluConvolution0", "ReluConvolution1", "Convolution2", "Convolution3", "ReluAddition4", "ReluConvolution5", "ReluConvolution6", "Convolution7", "Convolution8", "ReluAddition9", "ReluConvolution10", "ReluConvolution11", "ReluConvolution12", "ReluPooling13", "FullyConnected14"};
static int L3_input_layers[15] = {1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[15] = {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1};
static int branch_input[15] = {0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
static int branch_output[15] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[15] = {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[15] = {25262, 450182, 683582, 63704, 0, 677116, 904361, 1188042, 102910, 0, 1213171, 1789244, 2610756, 0, 66595};
static int weights_size[15] = {208, 3552, 5280, 480, 0, 5280, 7040, 9344, 896, 0, 9344, 14016, 20928, 0, 572};
static int activations_checksum[15][1] = {{
  2090675  },
{
  555965  },
{
  180473  },
{
  3118589  },
{
  -11083904  },
{
  185294  },
{
  263284  },
{
  77685  },
{
  1027889  },
{
  -16629277  },
{
  72845  },
{
  62815  },
{
  25667  },
{
  20830  },
{
  308  }
};
static int activations_size[15] = {16384, 65536, 24576, 65536, 98304, 24576, 24576, 8192, 24576, 32768, 8192, 8192, 3072, 3072, 48};
static int out_mult_vector[15] = {43, 55, 1, 1, 39, 49, 52, 1, 1, 37, 53, 43, 40, 2048, 1};
static int out_shift_vector[15] = {14, 14, 0, 0, 13, 14, 14, 0, 0, 14, 14, 14, 14, 11, 0};
static int activations_out_checksum[15][1] = {{
  555965 },
{
  180473 },
{
  15808985 },
{
  13478791 },
{
  185294 },
{
  263284 },
{
  77685 },
{
  4272953 },
{
  5199897 },
{
  72845 },
{
  62815 },
{
  25667 },
{
  20830 },
{
  308 },
{
  5949 }
};
static int activations_out_size[15] = {65536, 24576, 98304, 98304, 24576, 24576, 8192, 32768, 32768, 8192, 8192, 3072, 3072, 48, 44};
static int layer_with_weights[15] = {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1};
#endif

#endif  // __NETWORK_H__
