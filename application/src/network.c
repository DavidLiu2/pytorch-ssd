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
#include "ReluConvolution4.h"
#include "Convolution57.h"
#include "Addition58.h"
#include "Addition47.h"
#include "ReluConvolution56.h"
#include "Convolution70.h"
#include "ReluConvolution8.h"
#include "ReluConvolution49.h"
#include "ReluConvolution18.h"
#include "Convolution27.h"
#include "ReluConvolution30.h"
#include "ReluConvolution33.h"
#include "ReluConvolution60.h"
#include "Addition28.h"
#include "Addition17.h"
#include "Convolution31.h"
#include "ReluConvolution41.h"
#include "Convolution2.h"
#include "Convolution13.h"
#include "Convolution53.h"
#include "ReluConvolution51.h"
#include "Convolution24.h"
#include "Convolution50.h"
#include "Convolution42.h"
#include "ReluConvolution45.h"
#include "ReluConvolution52.h"
#include "Convolution6.h"
#include "Convolution16.h"
#include "Convolution61.h"
#include "ReluConvolution12.h"
#include "ReluConvolution48.h"
#include "ReluConvolution23.h"
#include "ReluConvolution62.h"
#include "ReluConvolution7.h"
#include "Convolution9.h"
#include "ReluConvolution22.h"
#include "Addition36.h"
#include "Convolution64.h"
#include "Convolution66.h"
#include "ReluConvolution25.h"
#include "ReluConvolution1.h"
#include "Convolution68.h"
#include "ReluConvolution19.h"
#include "ReluConvolution55.h"
#include "Convolution46.h"
#include "ReluConvolution0.h"
#include "ReluConvolution14.h"
#include "Addition10.h"
#include "ReluConvolution29.h"
#include "ReluConvolution37.h"
#include "Addition32.h"
#include "Addition43.h"
#include "ReluConvolution11.h"
#include "Convolution65.h"
#include "ReluConvolution44.h"
#include "Convolution20.h"
#include "Addition54.h"
#include "ReluConvolution38.h"
#include "Convolution39.h"
#include "ReluConvolution59.h"
#include "Convolution69.h"
#include "ReluConvolution26.h"
#include "Convolution63.h"
#include "ReluConvolution5.h"
#include "Addition3.h"
#include "ReluConvolution34.h"
#include "Convolution67.h"
#include "Addition21.h"
#include "ReluConvolution15.h"
#include "ReluConvolution40.h"
#include "Convolution35.h"


#define VERBOSE 1

#define L3_WEIGHTS_SIZE 4000000
#define L3_INPUT_SIZE 1500000
#define L3_OUTPUT_SIZE 1500000
static void *L3_weights = NULL;
static void *L3_input = NULL;
static void *L3_output = NULL;
int cycle_network_execution;
/* Moves the weights and the biases from hyperflash to hyperram */
void network_initialize() {

  L3_weights = ram_malloc(L3_WEIGHTS_SIZE);
  L3_input = ram_malloc(L3_INPUT_SIZE);
  L3_output = ram_malloc(L3_OUTPUT_SIZE);

#ifdef VERBOSE
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

struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  unsigned int args[4];
  args[0] = (unsigned int) l2_buffer;
  args[1] = (unsigned int) l2_buffer_size;
  args[2] = (unsigned int) l2_final_output;
  args[3] = (unsigned int) exec;
  args[4] = (unsigned int) initial_dir;
  // open cluster...
  pi_cluster_task(&cluster_task, network_run_cluster, args);
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return;
  // Then offload an entry point, this will get executed on the cluster controller
  cluster_task.stack_size = 3500;
  cluster_task.slave_stack_size = 3400;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  return (struct network_run_token) {
    .cluster_dev = cluster_dev
  };
}

void network_run_wait(struct network_run_token token)
{
  pi_cluster_close(&token.cluster_dev);
  print_perf("Final", cycle_network_execution, 21012800);
}

void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir)
{
  network_run_wait(network_run_async(l2_buffer, l2_buffer_size, l2_final_output, exec, initial_dir));
}

void network_run_cluster(void *args) {
  unsigned int * real_args = (unsigned int *) args;
  void * l2_buffer = (void *) real_args[0];
  size_t l2_buffer_size = (size_t) real_args[1];
  void * l2_final_output = (void *) real_args[2];
  int exec = (int) real_args[3];
  int dir = (int) real_args[4];
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

  int residual_number = 0;
  int bypass_dimension = 0;
  int perf_cyc = 0;
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
  for (int i = 0; i < 71; i++) {
/* MEMORY ALLOCATION
  - allocate memory if layer is executed from L3;
  - allocate weights
  - read weights
*/
    L2_output = dmalloc(activations_out_size[i], !dir);
    if (L3_input_layers[i] == 1)
      L2_input = dmalloc(activations_size[i], dir);

    if (layer_with_weights[i] == 1)
      L2_weights = dmalloc(weights_size[i], dir);

    if (allocate_layer[i] == 1)
      cl_ram_read(L2_weights, L3_weights_curr, weights_size[i]);

#ifdef VERBOSE
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


    // TODO: What error?
    // prevents error from compiler
    asm volatile("": : :"memory");
    unsigned int temp = L3_input;
    L3_input = L3_output;
    asm volatile("": : :"memory");
    L3_output = temp;
    asm volatile("": : :"memory");

#ifdef VERBOSE
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
    if (layer_with_weights[i] == 1)
      dfree(weights_size[i], dir);
    dfree(activations_size[i], dir);
    if (branch_input[i] == 1)
      dfree(bypass_dimension, dir);
    L2_input = L2_output;
    // Residual connections
    if (i < 70) {
      if (branch_input[i+1] == 1) {
        bypass_activations = dmalloc(bypass_dimension, !dir);
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
        L2_input = dmalloc(activations_size[i + 1], !dir);
        cl_ram_read(L2_input, layers_pointers[residual_number - 2], activations_size[i + 1]);
        cl_ram_free(layers_pointers[residual_number - 2], activations_size[i + 1]);
      }
      if (L3_output_layers[i] == 1)
        dfree(activations_out_size[i], !dir);
    }
    if (layer_with_weights[i])
       L3_weights_curr += L3_weights_size[weight_l_cnt++];
    dir = !dir;
  }

  //memcpy(L2_output, l2_final_output, activations_out_size[70]); // BUGGY!
  for (int i=0; i<activations_out_size[70]; i++)
    *((uint8_t*)(l2_final_output+i)) = *((uint8_t*)(L2_output+i));

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
