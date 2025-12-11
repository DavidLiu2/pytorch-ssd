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
#include "weights.h"
#include "net_utils.h"
#include "pmsis.h"
#include "network.h"
#include "directional_allocator.h"
#include "mem.h"
#include <string.h>
#include "Convolution49.h"
#include "Convolution34.h"
#include "Convolution53.h"
#include "Convolution5.h"
#include "Addition28.h"
#include "Convolution0.h"
#include "Convolution11.h"
#include "Convolution24.h"
#include "Convolution30.h"
#include "Convolution37.h"
#include "Convolution41.h"
#include "Convolution44.h"
#include "Convolution45.h"
#include "Convolution50.h"
#include "Addition21.h"
#include "Convolution31.h"
#include "Addition32.h"
#include "Convolution42.h"
#include "Convolution29.h"
#include "Convolution12.h"
#include "Convolution16.h"
#include "Convolution23.h"
#include "Convolution40.h"
#include "Convolution69.h"
#include "Convolution33.h"
#include "Addition17.h"
#include "Convolution39.h"
#include "Convolution26.h"
#include "Convolution14.h"
#include "Convolution13.h"
#include "Convolution48.h"
#include "Convolution59.h"
#include "Convolution7.h"
#include "Convolution55.h"
#include "Convolution9.h"
#include "Convolution61.h"
#include "Convolution38.h"
#include "Addition36.h"
#include "Convolution57.h"
#include "Convolution63.h"
#include "Convolution64.h"
#include "Convolution1.h"
#include "Convolution6.h"
#include "Convolution60.h"
#include "Addition10.h"
#include "Addition54.h"
#include "Convolution68.h"
#include "Convolution18.h"
#include "Convolution62.h"
#include "Convolution46.h"
#include "Convolution19.h"
#include "Addition58.h"
#include "Convolution8.h"
#include "Convolution20.h"
#include "Convolution67.h"
#include "Addition47.h"
#include "Convolution2.h"
#include "Convolution4.h"
#include "Addition3.h"
#include "Convolution15.h"
#include "Convolution22.h"
#include "Convolution52.h"
#include "Convolution70.h"
#include "Convolution35.h"
#include "Convolution56.h"
#include "Addition43.h"
#include "Convolution25.h"
#include "Convolution51.h"
#include "Convolution66.h"
#include "Convolution65.h"
#include "Convolution27.h"


#define VERBOSE 1

static void *L3_weights = NULL;
static void *L3_input = NULL;
static void *L3_output = NULL;
int cycle_network_execution;


void execute_layer_fork(void *args) {
  layer_args_t *layer_args = (layer_args_t *)args;
  if (pi_core_id() == 0) layer_args->L1_buffer = pmsis_l1_malloc(36700);

  switch (layer_args->layer_id)
  {
    case 0:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution0, args);
      break;
    case 1:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution1, args);
      break;
    case 2:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution2, args);
      break;
    case 3:
      pi_cl_team_fork(NUM_CORES, (void *)Addition3, args);
      break;
    case 4:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution4, args);
      break;
    case 5:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution5, args);
      break;
    case 6:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution6, args);
      break;
    case 7:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution7, args);
      break;
    case 8:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution8, args);
      break;
    case 9:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution9, args);
      break;
    case 10:
      pi_cl_team_fork(NUM_CORES, (void *)Addition10, args);
      break;
    case 11:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution11, args);
      break;
    case 12:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution12, args);
      break;
    case 13:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution13, args);
      break;
    case 14:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution14, args);
      break;
    case 15:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution15, args);
      break;
    case 16:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution16, args);
      break;
    case 17:
      pi_cl_team_fork(NUM_CORES, (void *)Addition17, args);
      break;
    case 18:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution18, args);
      break;
    case 19:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution19, args);
      break;
    case 20:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution20, args);
      break;
    case 21:
      pi_cl_team_fork(NUM_CORES, (void *)Addition21, args);
      break;
    case 22:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution22, args);
      break;
    case 23:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution23, args);
      break;
    case 24:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution24, args);
      break;
    case 25:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution25, args);
      break;
    case 26:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution26, args);
      break;
    case 27:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution27, args);
      break;
    case 28:
      pi_cl_team_fork(NUM_CORES, (void *)Addition28, args);
      break;
    case 29:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution29, args);
      break;
    case 30:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution30, args);
      break;
    case 31:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution31, args);
      break;
    case 32:
      pi_cl_team_fork(NUM_CORES, (void *)Addition32, args);
      break;
    case 33:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution33, args);
      break;
    case 34:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution34, args);
      break;
    case 35:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution35, args);
      break;
    case 36:
      pi_cl_team_fork(NUM_CORES, (void *)Addition36, args);
      break;
    case 37:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution37, args);
      break;
    case 38:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution38, args);
      break;
    case 39:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution39, args);
      break;
    case 40:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution40, args);
      break;
    case 41:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution41, args);
      break;
    case 42:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution42, args);
      break;
    case 43:
      pi_cl_team_fork(NUM_CORES, (void *)Addition43, args);
      break;
    case 44:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution44, args);
      break;
    case 45:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution45, args);
      break;
    case 46:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution46, args);
      break;
    case 47:
      pi_cl_team_fork(NUM_CORES, (void *)Addition47, args);
      break;
    case 48:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution48, args);
      break;
    case 49:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution49, args);
      break;
    case 50:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution50, args);
      break;
    case 51:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution51, args);
      break;
    case 52:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution52, args);
      break;
    case 53:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution53, args);
      break;
    case 54:
      pi_cl_team_fork(NUM_CORES, (void *)Addition54, args);
      break;
    case 55:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution55, args);
      break;
    case 56:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution56, args);
      break;
    case 57:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution57, args);
      break;
    case 58:
      pi_cl_team_fork(NUM_CORES, (void *)Addition58, args);
      break;
    case 59:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution59, args);
      break;
    case 60:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution60, args);
      break;
    case 61:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution61, args);
      break;
    case 62:
      pi_cl_team_fork(NUM_CORES, (void *)Convolution62, args);
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

struct network_run_token network_run_async(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir, void *L2_input_h)
{
  struct pi_device cluster_dev = {0};
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task = {0};
  // First open the cluster
  pi_cluster_conf_init(&conf);
  conf.id=0;
  unsigned int args[5];
  args[0] = (unsigned int) l2_buffer;
  args[1] = (unsigned int) l2_buffer_size;
  args[2] = (unsigned int) l2_final_output;
  args[3] = (unsigned int) exec;
  args[4] = (unsigned int) initial_dir;
  args[5] = (unsigned int) L2_input_h;
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

void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int initial_dir, void *L2_input_h)
{
  network_run_wait(network_run_async(l2_buffer, l2_buffer_size, l2_final_output, exec, initial_dir, L2_input_h));
}

void network_run_cluster(void *args) {
  unsigned int * real_args = (unsigned int *) args;
  void * l2_buffer = (void *) real_args[0];
  size_t l2_buffer_size = (size_t) real_args[1];
  void * l2_final_output = (void *) real_args[2];
  int exec = (int) real_args[3];
  int dir = (int) real_args[4];
  void * L2_input_h = (void *)real_args[5];
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
  int left_branch_nodes = 0, right_branch_nodes = 0;
  int z = 0;
  int end_left = 0;
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
  L2_input = L2_input_h;
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
    L2_weights = Weights_name[i];

#ifdef VERBOSE
    if (i == 0 || branch_change[i-1] == 0) {
      checksum("L2 input", L2_input, activations_size[i], activations_checksum[i][exec]);
      if (layer_with_weights[i])
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
      checksum(i + 1 < 71 ? "L2 output" : "final output",
               L2_output, activations_out_size[i], activations_out_checksum[i][exec]);
    printf("\n");
#endif

    // Free memory
    if (branch_input[i] == 1)
      dfree(bypass_dimension, dir);
    L2_input = L2_output;
    if  (branch_output[i]==1)
      {
        bypass_activations = L2_output;
        bypass_dimension = activations_out_size[i];
      }

    if (i > 0 && branch_output[i-1] == 0 && branch_change[i-1] == 0)
      dfree(activations_size[i], dir);
    // Residual connections
    if (i < 70) {

      if  (branch_output[i]==1)
      {
        left_branch_nodes = 0;
        right_branch_nodes = 0;
        z = i+1;
        end_left = 0;
        while (branch_input[z] == 0)
        {
          if (end_left == 0)
            left_branch_nodes+=1;
          else
            right_branch_nodes+=1;
          if (branch_change[z] == 1)
            end_left = 1;
          z+=1;
        }
        if ((left_branch_nodes % 2 == 1) && (right_branch_nodes == 0))
          dir = !dir;
        if ((left_branch_nodes % 2 == 0) && (right_branch_nodes > 0))
          dir = !dir;
      }

      if  (branch_change[i]==1)
      {
        L2_input = bypass_activations;
        bypass_activations = L2_output;
        bypass_dimension = activations_out_size[i];
        if (right_branch_nodes % 2 == 1)
          dir = !dir;
      }
    }
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
