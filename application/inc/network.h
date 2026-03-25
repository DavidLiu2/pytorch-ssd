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
  "ReluConvolution0_weights.hex", "Convolution1_weights.hex", "ReluConvolution2_weights.hex", "Convolution3_weights.hex", "ReluConvolution5_weights.hex", "Convolution6_weights.hex", "Convolution8_weights.hex", "ReluConvolution9_weights.hex", "Convolution10_weights.hex", "ReluConvolution12_weights.hex", "Convolution13_weights.hex", "Convolution15_weights.hex", "ReluConvolution16_weights.hex", "Convolution17_weights.hex", "ReluConvolution19_weights.hex", "Convolution20_weights.hex", "Convolution22_weights.hex", "ReluConvolution23_weights.hex", "Convolution24_weights.hex", "ReluConvolution26_weights.hex", "Convolution27_weights.hex", "FullyConnected30_weights.hex"
};
static int L3_weights_size[22];
static int layers_pointers[31];
static char * Layers_name[31] = {"ReluConvolution0", "Convolution1", "ReluConvolution2", "Convolution3", "ReluQAddition4", "ReluConvolution5", "Convolution6", "ReluQAddition7", "Convolution8", "ReluConvolution9", "Convolution10", "ReluQAddition11", "ReluConvolution12", "Convolution13", "ReluQAddition14", "Convolution15", "ReluConvolution16", "Convolution17", "ReluQAddition18", "ReluConvolution19", "Convolution20", "ReluQAddition21", "Convolution22", "ReluConvolution23", "Convolution24", "ReluQAddition25", "ReluConvolution26", "Convolution27", "ReluQAddition28", "ReluPooling29", "FullyConnected30"};
static int L3_input_layers[31] = {1,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int L3_output_layers[31] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int allocate_layer[31] = {1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1};
static int branch_input[31] = {0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0};
static int branch_output[31] = {1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0};
static int branch_change[31] = {0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[31] = {17604, 58200, 476581, 687530, 0, 700230, 662219, 0, 98438, 933060, 1243134, 0, 1254707, 1226619, 0, 271505, 2435379, 4946877, 0, 5000696, 4882347, 0, 651626, 6103168, 7539176, 0, 7712383, 7297848, 0, 0, 29305};
static int weights_size[31] = {208, 480, 3552, 5280, 0, 5280, 5280, 0, 896, 7040, 9344, 0, 9344, 9344, 0, 2304, 18688, 37120, 0, 37120, 37120, 0, 5440, 46400, 57920, 0, 57920, 57920, 0, 0, 252};
static int activations_checksum[31][1] = {{
  2090675  },
{
  7973547  },
{
  3094112  },
{
  1118191  },
{
  3135389  },
{
  1137073  },
{
  821479  },
{
  3154621  },
{
  2058979  },
{
  1027967  },
{
  709406  },
{
  1046459  },
{
  599099  },
{
  312309  },
{
  1043768  },
{
  593570  },
{
  518806  },
{
  358395  },
{
  522460  },
{
  229595  },
{
  221156  },
{
  531897  },
{
  213727  },
{
  160764  },
{
  73647  },
{
  161916  },
{
  108933  },
{
  49470  },
{
  162434  },
{
  157447  },
{
  9805  }
};
static int activations_size[31] = {16384, 65536, 65536, 24576, 24576, 24576, 24576, 24576, 24576, 24576, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 1280, 1280, 1280, 1280, 1280, 1280, 80};
static int out_mult_vector[31] = {36, 1, 45, 1, 54, 43, 1, 39, 1, 56, 1, 39, 44, 1, 36, 1, 37, 1, 47, 41, 1, 43, 1, 52, 1, 46, 41, 1, 37, 512, 1};
static int out_shift_vector[31] = {7, 0, 7, 0, 8, 7, 0, 8, 0, 7, 0, 7, 6, 0, 7, 0, 6, 0, 7, 6, 0, 7, 0, 6, 0, 7, 6, 0, 7, 9, 0};
static int activations_out_checksum[31][1] = {{
  7973547 },
{
  15654307 },
{
  1118191 },
{
  14884838 },
{
  1137073 },
{
  821479 },
{
  14535565 },
{
  2058979 },
{
  4445358 },
{
  709406 },
{
  4983137 },
{
  599099 },
{
  312309 },
{
  5054214 },
{
  593570 },
{
  2282691 },
{
  358395 },
{
  2659996 },
{
  229595 },
{
  221156 },
{
  2695619 },
{
  213727 },
{
  686436 },
{
  73647 },
{
  738168 },
{
  108933 },
{
  49470 },
{
  660613 },
{
  157447 },
{
  9805 },
{
  752 }
};
static int activations_out_size[31] = {65536, 98304, 24576, 98304, 24576, 24576, 98304, 24576, 32768, 8192, 32768, 8192, 8192, 32768, 8192, 16384, 4096, 16384, 4096, 4096, 16384, 4096, 5120, 1280, 5120, 1280, 1280, 5120, 1280, 80, 12};
static int layer_with_weights[31] = {1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1};
#endif

#endif  // __NETWORK_H__
