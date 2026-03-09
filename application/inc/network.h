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
static int weights_checksum[71] = {25095, 9672, 7591, 0, 52828, 49904, 46921, 51979, 56434, 46610, 0, 48574, 53496, 48664, 49960, 52421, 49670, 0, 49785, 55967, 49055, 0, 48298, 57409, 49276, 47279, 59030, 51521, 0, 46430, 59731, 48499, 0, 46109, 62498, 47112, 0, 46361, 57430, 95897, 192588, 112363, 194432, 0, 191242, 113895, 197012, 0, 196588, 118528, 196914, 195222, 115450, 196382, 0, 198881, 113100, 197091, 0, 196625, 115754, 386107, 524431, 295972, 276954, 561948, 4722122, 150592, 149100, 303755, 2304055};
static int weights_size[71] = {216, 72, 64, 0, 384, 432, 384, 384, 432, 384, 0, 384, 432, 384, 384, 432, 384, 0, 384, 432, 384, 0, 384, 432, 384, 384, 432, 384, 0, 384, 432, 384, 0, 384, 432, 384, 0, 384, 432, 768, 1536, 864, 1536, 0, 1536, 864, 1536, 0, 1536, 864, 1536, 1536, 864, 1536, 0, 1536, 864, 1536, 0, 1536, 864, 3072, 4096, 2432, 2432, 4736, 36992, 1216, 1216, 2368, 18496};
static int activations_checksum[71][1] = {{
  9776730  },
{
  7162791  },
{
  3493520  },
{
  37321647  },
{
  7802710  },
{
  41626646  },
{
  6886646  },
{
  1670650  },
{
  9545415  },
{
  4203254  },
{
  -220570042  },
{
  1676864  },
{
  8901030  },
{
  1535113  },
{
  392215  },
{
  2446573  },
{
  1384674  },
{
  28088464  },
{
  409255  },
{
  2741760  },
{
  1687574  },
{
  -38411490  },
{
  402885  },
{
  2754510  },
{
  443555  },
{
  99271  },
{
  565590  },
{
  410517  },
{
  -18465363  },
{
  102004  },
{
  544170  },
{
  505743  },
{
  5821074  },
{
  99590  },
{
  639795  },
{
  375518  },
{
  7109607  },
{
  100845  },
{
  528615  },
{
  537990  },
{
  204789  },
{
  1235475  },
{
  905718  },
{
  31634560  },
{
  204149  },
{
  1228845  },
{
  726287  },
{
  -16659297  },
{
  203540  },
{
  1221793  },
{
  198732  },
{
  49036  },
{
  261327  },
{
  210922  },
{
  6969704  },
{
  50164  },
{
  302685  },
{
  252311  },
{
  353015  },
{
  50667  },
{
  303960  },
{
  260486  },
{
  103665  },
{
  327165  },
{
  6510034  },
{
  1602517  },
{
  405173  },
{
  8618304  },
{
  3233890  },
{
  811326  },
{
  202401  }
};
static int activations_size[71] = {76800, 51200, 51200, 204800, 51200, 307200, 76800, 12800, 76800, 76800, 51200, 12800, 76800, 19200, 3200, 19200, 19200, 12800, 3200, 19200, 19200, 12800, 3200, 19200, 4800, 800, 4800, 4800, 3200, 800, 4800, 4800, 3200, 800, 4800, 4800, 3200, 800, 4800, 4800, 1600, 9600, 9600, 6400, 1600, 9600, 9600, 6400, 1600, 9600, 2400, 400, 2400, 2400, 1600, 400, 2400, 2400, 1600, 400, 2400, 2400, 800, 12800, 3200, 1600, 3200, 12800, 3200, 1600, 3200};
static int out_mult_vector[71] = {49, 62, 1, 1, 48, 46, 1, 42, 35, 1, 1, 34, 34, 1, 45, 33, 1, 1, 50, 37, 1, 1, 55, 38, 1, 54, 34, 1, 1, 53, 42, 1, 1, 56, 36, 1, 1, 58, 42, 1, 47, 36, 1, 1, 41, 37, 1, 1, 44, 43, 1, 52, 40, 1, 1, 54, 39, 1, 1, 53, 34, 1, 51, 1, 1, 1, 1, 1, 1, 1, 1};
static int out_shift_vector[71] = {7, 7, 0, 0, 7, 8, 0, 7, 8, 0, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0};
static int activations_out_checksum[71][1] = {{
  7162791 },
{
  3493520 },
{
  23175221 },
{
  23154151 },
{
  41626646 },
{
  6886646 },
{
  5953956 },
{
  9545415 },
{
  4203254 },
{
  8329505 },
{
  6142988 },
{
  8901030 },
{
  1535113 },
{
  1421544 },
{
  2446573 },
{
  1384674 },
{
  1533410 },
{
  1336153 },
{
  2741760 },
{
  1687574 },
{
  1750566 },
{
  1458483 },
{
  2754510 },
{
  443555 },
{
  449678 },
{
  565590 },
{
  410517 },
{
  470441 },
{
  488468 },
{
  544170 },
{
  505743 },
{
  385849 },
{
  464893 },
{
  639795 },
{
  375518 },
{
  381783 },
{
  449001 },
{
  528615 },
{
  537990 },
{
  805916 },
{
  1235475 },
{
  905718 },
{
  690874 },
{
  722896 },
{
  1228845 },
{
  726287 },
{
  890749 },
{
  789693 },
{
  1221793 },
{
  198732 },
{
  199313 },
{
  261327 },
{
  210922 },
{
  182016 },
{
  181985 },
{
  302685 },
{
  252311 },
{
  194589 },
{
  188473 },
{
  303960 },
{
  260486 },
{
  399842 },
{
  327165 },
{
  30118921 },
{
  7172802 },
{
  1671180 },
{
  39314577 },
{
  14549750 },
{
  3273626 },
{
  759574 },
{
  18785620 }
};
static int activations_out_size[71] = {51200, 51200, 204800, 204800, 307200, 76800, 51200, 76800, 76800, 51200, 51200, 76800, 19200, 12800, 19200, 19200, 12800, 12800, 19200, 19200, 12800, 12800, 19200, 4800, 3200, 4800, 4800, 3200, 3200, 4800, 4800, 3200, 3200, 4800, 4800, 3200, 3200, 4800, 4800, 6400, 9600, 9600, 6400, 6400, 9600, 9600, 6400, 6400, 9600, 2400, 1600, 2400, 2400, 1600, 1600, 2400, 2400, 1600, 1600, 2400, 2400, 3200, 3200, 204800, 51200, 12800, 3200, 102400, 25600, 6400, 1600};
static int layer_with_weights[71] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#endif

#endif  // __NETWORK_H__
