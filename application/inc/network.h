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

#include "weights_definition.h"
#include <stddef.h>
#include "pmsis.h"


struct network_run_token {
  struct pi_device cluster_dev;
};


void network_run_cluster(void * args);
struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir, void *L2_input_h);
void network_run_wait(struct network_run_token token);
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir, void *L2_input_h);
void execute_layer_fork(void *arg);


#ifdef DEFINE_CONSTANTS
static char * Layers_name[71] = {"Convolution0", "Convolution1", "Convolution2", "Addition3", "Convolution4", "Convolution5", "Convolution6", "Convolution7", "Convolution8", "Convolution9", "Addition10", "Convolution11", "Convolution12", "Convolution13", "Convolution14", "Convolution15", "Convolution16", "Addition17", "Convolution18", "Convolution19", "Convolution20", "Addition21", "Convolution22", "Convolution23", "Convolution24", "Convolution25", "Convolution26", "Convolution27", "Addition28", "Convolution29", "Convolution30", "Convolution31", "Addition32", "Convolution33", "Convolution34", "Convolution35", "Addition36", "Convolution37", "Convolution38", "Convolution39", "Convolution40", "Convolution41", "Convolution42", "Addition43", "Convolution44", "Convolution45", "Convolution46", "Addition47", "Convolution48", "Convolution49", "Convolution50", "Convolution51", "Convolution52", "Convolution53", "Addition54", "Convolution55", "Convolution56", "Convolution57", "Addition58", "Convolution59", "Convolution60", "Convolution61", "Convolution62", "Convolution63", "Convolution64", "Convolution65", "Convolution66", "Convolution67", "Convolution68", "Convolution69", "Convolution70"};
static char *Weights_name[71] = {Weights_Convolution0, Weights_Convolution1, Weights_Convolution2, "None", Weights_Convolution4, Weights_Convolution5, Weights_Convolution6, Weights_Convolution7, Weights_Convolution8, Weights_Convolution9, "None", Weights_Convolution11, Weights_Convolution12, Weights_Convolution13, Weights_Convolution14, Weights_Convolution15, Weights_Convolution16, "None", Weights_Convolution18, Weights_Convolution19, Weights_Convolution20, "None", Weights_Convolution22, Weights_Convolution23, Weights_Convolution24, Weights_Convolution25, Weights_Convolution26, Weights_Convolution27, "None", Weights_Convolution29, Weights_Convolution30, Weights_Convolution31, "None", Weights_Convolution33, Weights_Convolution34, Weights_Convolution35, "None", Weights_Convolution37, Weights_Convolution38, Weights_Convolution39, Weights_Convolution40, Weights_Convolution41, Weights_Convolution42, "None", Weights_Convolution44, Weights_Convolution45, Weights_Convolution46, "None", Weights_Convolution48, Weights_Convolution49, Weights_Convolution50, Weights_Convolution51, Weights_Convolution52, Weights_Convolution53, "None", Weights_Convolution55, Weights_Convolution56, Weights_Convolution57, "None", Weights_Convolution59, Weights_Convolution60, Weights_Convolution61, Weights_Convolution62, Weights_Convolution63, Weights_Convolution64, Weights_Convolution65, Weights_Convolution66, Weights_Convolution67, Weights_Convolution68, Weights_Convolution69, Weights_Convolution70};
static int branch_input[71] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_output[71] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
static int branch_change[71] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_checksum[71] = {0, 256, 2562, 0, 255, 2, 0, 504, 0, 0, 0, 511, 0, 0, 763, 0, 0, 0, 511, 0, 0, 0, 511, 0, 0, 768, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int weights_size[71] = {248, 1664, 96, 0, 576, 9984, 416, 576, 9984, 416, 0, 576, 9984, 416, 576, 9984, 416, 0, 576, 9984, 416, 0, 576, 9984, 416, 576, 9984, 416, 0, 576, 9984, 416, 0, 576, 9984, 416, 0, 576, 9984, 832, 1920, 19968, 1600, 0, 1920, 19968, 1600, 0, 1920, 19968, 1600, 1920, 19968, 1600, 0, 1920, 19968, 1600, 0, 1920, 19968, 3200, 4608, 2432, 2432, 4736, 36992, 1216, 1216, 2368, 18496};
static int activations_checksum[71][1] = {{
  9767697  },
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
static int activations_size[71] = {76800, 51200, 51200, 204800, 51200, 307200, 76800, 12800, 76800, 76800, 51200, 12800, 76800, 19200, 3200, 19200, 19200, 12800, 3200, 19200, 19200, 12800, 3200, 19200, 4800, 800, 4800, 4800, 3200, 800, 4800, 4800, 3200, 800, 4800, 4800, 3200, 800, 4800, 4800, 1600, 9600, 9600, 6400, 1600, 9600, 9600, 6400, 1600, 9600, 2400, 400, 2400, 2400, 1600, 400, 2400, 2400, 1600, 400, 2400, 2400, 800, 12800, 3200, 1600, 3200, 12800, 3200, 1600, 3200};
static int out_mult_vector[71] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static int out_shift_vector[71] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
static int activations_out_checksum[71][1] = {{
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
static int activations_out_size[71] = {204800, 204800, 204800, 204800, 1228800, 307200, 51200, 307200, 307200, 51200, 51200, 307200, 76800, 12800, 76800, 76800, 12800, 12800, 76800, 76800, 12800, 12800, 76800, 19200, 3200, 19200, 19200, 3200, 3200, 19200, 19200, 3200, 3200, 19200, 19200, 3200, 3200, 19200, 19200, 6400, 38400, 38400, 6400, 6400, 38400, 38400, 6400, 6400, 38400, 9600, 1600, 9600, 9600, 1600, 1600, 9600, 9600, 1600, 1600, 9600, 9600, 3200, 12800, 204800, 51200, 12800, 3200, 102400, 25600, 6400, 1600};
static int layer_with_weights[71] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
#endif

#endif  // __NETWORK_H__
