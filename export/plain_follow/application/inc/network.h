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
  "ReluConvolution0_weights.hex", "ReluConvolution1_weights.hex", "ReluConvolution2_weights.hex", "ReluConvolution3_weights.hex", "ReluConvolution4_weights.hex", "ReluConvolution5_weights.hex", "ReluConvolution6_weights.hex", "FullyConnected8_weights.hex"
};
static int L3_weights_size[8];
static int layers_pointers[9];
static char * Layers_name[9] = {"ReluConvolution0", "ReluConvolution1", "ReluConvolution2", "ReluConvolution3", "ReluConvolution4", "ReluConvolution5", "ReluConvolution6", "ReluPooling7", "FullyConnected8"};
static int L3_input_layers[9] = {1,
0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[9] = {1, 1, 1, 1, 1, 1, 1, 0, 1};
static int branch_input[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[9] = {18112, 453535, 670150, 918495, 1234819, 1807855, 2578266, 0, 88415};
static int weights_size[9] = {144, 3456, 5184, 6912, 9216, 13824, 20736, 0, 728};
static int activations_checksum[9][1] = {{
  2090675  },
{
  0  },
{
  0  },
{
  0  },
{
  0  },
{
  0  },
{
  0  },
{
  0  },
{
  0  }
};
static int activations_size[9] = {16384, 65536, 24576, 24576, 8192, 8192, 3072, 3072, 48};
static int out_mult_vector[9] = {38, 47, 46, 39, 31, 30, 23, 2048, 1};
static int out_shift_vector[9] = {24, 24, 24, 24, 24, 24, 24, 11, 0};
static int activations_out_checksum[9][1] = {{
  0 },
{
  0 },
{
  0 },
{
  0 },
{
  0 },
{
  0 },
{
  0 },
{
  0 },
{
  0 }
};
static int activations_out_size[9] = {65536, 24576, 24576, 8192, 8192, 3072, 3072, 48, 56};
static int layer_with_weights[9] = {1, 1, 1, 1, 1, 1, 1, 0, 1};
#endif

#endif  // __NETWORK_H__
