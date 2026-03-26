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
static int weights_checksum[31] = {17601, 58205, 476536, 687532, 0, 700279, 661201, 0, 98456, 933077, 1242230, 0, 1254714, 1226648, 0, 271520, 2435398, 4947524, 0, 5000776, 4882357, 0, 649601, 6103173, 7538329, 0, 7711375, 7297951, 0, 0, 29309};
static int weights_size[31] = {208, 480, 3552, 5280, 0, 5280, 5280, 0, 896, 7040, 9344, 0, 9344, 9344, 0, 2304, 18688, 37120, 0, 37120, 37120, 0, 5440, 46400, 57920, 0, 57920, 57920, 0, 0, 252};
static int activations_checksum[31][1] = {{
  2090675  },
{
  1405737  },
{
  3135075  },
{
  247944  },
{
  3114972  },
{
  1851119  },
{
  422551  },
{
  3144884  },
{
  1395485  },
{
  1047636  },
{
  262818  },
{
  1053911  },
{
  839601  },
{
  258919  },
{
  1046941  },
{
  716372  },
{
  530913  },
{
  164379  },
{
  522714  },
{
  419991  },
{
  185700  },
{
  522071  },
{
  273417  },
{
  163776  },
{
  51937  },
{
  159470  },
{
  138651  },
{
  30761  },
{
  165984  },
{
  105288  },
{
  6541  }
};
static int activations_size[31] = {16384, 65536, 65536, 24576, 24576, 24576, 24576, 24576, 24576, 24576, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 1280, 1280, 1280, 1280, 1280, 1280, 80};
static int out_mult_vector[31] = {61, 1, 38, 1, 63, 59, 1, 62, 1, 43, 1, 42, 51, 1, 32, 1, 44, 1, 42, 43, 1, 59, 1, 36, 1, 34, 36, 1, 45, 512, 1};
static int out_shift_vector[31] = {13, 0, 13, 0, 5, 14, 0, 5, 0, 14, 0, 4, 14, 0, 4, 0, 14, 0, 4, 14, 0, 5, 0, 14, 0, 4, 14, 0, 5, 9, 0};
static int activations_out_checksum[31][1] = {{
  1405737 },
{
  15962033 },
{
  247944 },
{
  15292674 },
{
  1851119 },
{
  422551 },
{
  15525496 },
{
  1395485 },
{
  4661316 },
{
  262818 },
{
  5211580 },
{
  839601 },
{
  258919 },
{
  4819230 },
{
  716372 },
{
  2313581 },
{
  164379 },
{
  2463400 },
{
  419991 },
{
  185700 },
{
  2562013 },
{
  273417 },
{
  702873 },
{
  51937 },
{
  704664 },
{
  138651 },
{
  30761 },
{
  617741 },
{
  105288 },
{
  6541 },
{
  767 }
};
static int activations_out_size[31] = {65536, 98304, 24576, 98304, 24576, 24576, 98304, 24576, 32768, 8192, 32768, 8192, 8192, 32768, 8192, 16384, 4096, 16384, 4096, 4096, 16384, 4096, 5120, 1280, 5120, 1280, 1280, 5120, 1280, 80, 12};
static int layer_with_weights[31] = {1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1};
#endif

#endif  // __NETWORK_H__
