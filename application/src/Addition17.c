/*
 * add_layer_template.c
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
// first_layer                    0
// ULTRA_VERBOSE                  False
// verbose_log                    
// node                           <dory.Parsers.HW_node.HW_node object at 0x7d5d187041f0>
// sdk                            gap_sdk
// number_of_clusters             1
// optional_type                  8bit
// func_name                      Addition17
// flag_DW                        0
// optional                       Add
// FLAG_BATCHNORM                 0
// has_bias                       0
// FLAG_RELU                      0
// type                           uint8_t
// conv_overlap1                  0
// conv_overlap2                  0
// padding_top                    0
// padding_bottom                 0
// padding_left                   0
// padding_right                  0
// stride                         1
// g                              1
// nif                            8
// data_type_x2                   uint
// x_data_size_byte2              32
// inmul1                         1
// inadd1                         0
// inshift1                       0
// inmul2                         1
// inadd2                         0
// inshift2                       0
// outmul                         1
// outadd                         0
// outshift                       0
// out_mul                        1
// out_add                        0
// out_shift                      0
// data_type_x                    uint
// data_type_y                    uint
// data_type_activations          int
// data_type_weights              int
// nof                            8
// factor                         1
// double_buffering               1
// x_h                            20
// x_w                            20
// x_data_size_byte               32
// x_tile_size_nif                8
// x_tile_size_h                  19
// x_tile_size_w                  20
// x_tile_size_byte               12160
// x_tile_size_nif_byte           32
// x_stride_w_byte                640
// x_stride_c_byte                32
// y_h                            20
// y_w                            20
// y_data_size_byte               32
// act_dim_bit                    None
// y_tile_size_nof                8
// y_tile_size_h                  19
// y_tile_size_w                  20
// y_tile_size_byte               12160
// y_stride_w_byte                640
// y_stride_c_byte                32
// y_tile_size_nof_byte           32
// tile_dim_h                     2
// tile_dim_w                     1
// tile_dim_nof                   1
// tile_dim_nif                   1
// tile_n_in_last                 8
// fs1                            1
// fs2                            1
// W_data_size_byte               None
// b_data_size_byte               32
// W_tile_size_nof                8
// b_size_byte                    0
// W_tile_size_nif                8
// W_tile_size_nif_last           8
// k_tile_size_byte               0
// lambda_tile_size_byte          0
// k_size_byte                    0
// lambda_size_byte               0
// k_tile_size_byte_transfer      0
// lambda_tile_size_byte_transfer 0
// bias_tile_size_byte            0
// l1_x_offset                    0
// l1_y_offset                    12168
// l1_x2_offset                   24336
// y_tile_size_nof_last           8
// y_tile_size_h_last             1
// y_tile_size_w_last             20
// y_length_nof_byte_last         32
// x_tile_size_nif_last           8
// x_tile_size_nif_byte_last      32
// x_tile_size_h_last             1
// x_tile_size_w_last             20


#include "Addition17.h"
#include "pulp.h"
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"


#ifdef SINGLE_CORE_DMA
L1_DATA static uint32_t dory_dma_channel = 0;
#endif


void Addition17(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int out_mult_in =(unsigned int)  real_arg[9];

  unsigned short x_tile_size_nif;
  unsigned short  x_tile_size_h;
  unsigned short  x_tile_size_w;
  unsigned short  x_tile_size_byte;
  unsigned short  x_length_h_px;
  unsigned short  x_length_nif_byte;
  int pad_offset_h, pad_offset_w;

  uint8_t *x;
  uint8_t *x2;
  uint8_t *y;
  int y_tile_size_nof;
  int y_tile_size_h;
  int y_tile_size_w;
  int y_tile_size_byte;
  int y_length_h_px;
  int y_length_nof_byte;
  // copy first tiles
  //l2_x has input activations
#ifndef SINGLE_CORE_DMA
  uint32_t dory_dma_channel = dory_dma_allocate();
#else
  if (pi_core_id() == 0)
    dory_dma_channel = dory_dma_allocate();
#endif
  volatile DMA_copy DMA_copy_x, DMA_copy_x2, DMA_copy_y;

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = 640;
  DMA_copy_x.stride_1d = 32;
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dory_dma_channel;

  DMA_copy_x2.hwc_to_chw = 0;
  DMA_copy_x2.stride_2d = 640;
  DMA_copy_x2.stride_1d = 32;
  DMA_copy_x2.dir = 1;
  DMA_copy_x2.tid = dory_dma_channel;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = 640;
  DMA_copy_y.stride_1d = 32;
  DMA_copy_y.dir = 0;
  DMA_copy_y.tid = dory_dma_channel;

  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;


  // last-tile flags
  int last_nof, last_nif, last_h, last_w;
  int iter;
  // tile loop nest
  for(iter=0; iter<1*2*1; iter++) {

    last_nof = (_i_nof_load+1 == 1) ? 1 : 0;
    last_nif = (_i_nof_load+1 == 1) ? 1 : 0;
    last_h = (_i_h_load+1 == 2) ? 1 : 0;
    last_w = (_i_w_load+1 == 1) ? 1 : 0;

    x_tile_size_nif = (last_nif) ? 8 : 8;
    x_tile_size_h   = (last_h)   ? 1 : 19;
    x_tile_size_w   = (last_w)   ? 20 : 20;
    x_tile_size_byte = x_tile_size_nif*x_tile_size_h*x_tile_size_w*32/8;
    x_length_nif_byte = (last_nif)   ? 32 : 32;
    // additionally overlap by padding for the first tile after a border one
    //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding

    DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, 19, 20, 8, 20, 8,  0, 0,0, 0, 0, 0, 32);
    DMA_copy_x.loc = (l1_buffer + 0);
    DMA_copy_x.number_of_2d_copies = x_tile_size_h;
    DMA_copy_x.number_of_1d_copies = x_tile_size_w;
    DMA_copy_x.length_1d_copy = x_length_nif_byte;
    dory_dma_memcpy_async(&DMA_copy_x);
    dory_dma_barrier(&DMA_copy_x);

    DMA_copy_x2.ext = dory_get_tile_3d(l2_x2, _i_h_load, _i_w_load, _i_nif_load, 19, 20, 8, 20, 8,  0, 0,0, 0, 0, 0, 32);
    DMA_copy_x2.loc = (l1_buffer + 24336);
    DMA_copy_x2.number_of_2d_copies = x_tile_size_h;
    DMA_copy_x2.number_of_1d_copies = x_tile_size_w;
    DMA_copy_x2.length_1d_copy = x_length_nif_byte;
    dory_dma_memcpy_async(&DMA_copy_x2);
    dory_dma_barrier(&DMA_copy_x2);

    y_tile_size_h   = (last_h)   ? 1 : 19;
    y_tile_size_w   = (last_w)   ? 20 : 20;

    x = (uint8_t *) (l1_buffer + 0);
    x2 = (uint8_t *) (l1_buffer + 24336);
    y = (uint8_t *) (l1_buffer + 12168);

    y_tile_size_nof = (last_nof) ? 8 : 8;
    y_tile_size_h   = (last_h)   ? 1 : 19;
    y_tile_size_w   = (last_w)   ? 20 : 20;
    y_tile_size_byte = y_tile_size_nof*y_tile_size_h*y_tile_size_w*32/8;
    y_length_nof_byte = (last_nof)   ? 32 : 32;
    asm volatile("": : :"memory");
    pi_cl_team_barrier(0);
    pulp_nn_add(
      x,
      x2,
      y,
      1,
      1,
      0,
      x_tile_size_w,
      x_tile_size_h,
      x_tile_size_nif
      );

    pi_cl_team_barrier(0);
    // wait for DMA write
    // copying output back to L2
    DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_load, _i_w_load, _i_nof_load, 19, 20, 8, 20, 8, 0, 0, 0, 0, 0, 0, 32);
    DMA_copy_y.loc = (l1_buffer + 12168);
    DMA_copy_y.number_of_2d_copies = y_tile_size_h;
    DMA_copy_y.number_of_1d_copies = y_tile_size_w;
    DMA_copy_y.length_1d_copy = y_length_nof_byte;
    dory_dma_memcpy_async(&DMA_copy_y); 
    dory_dma_barrier(&DMA_copy_y); 

    // loop nest is nof,h,w,(nif=0)
    _i_w_load += 1;
    if(_i_w_load==1) 
    {
      _i_w_load = 0;
      _i_h_load += 1;
      if(_i_h_load==2) 
      {
        _i_h_load = 0;
        _i_nif_load += 1;
        _i_nof_load += 1;
      }
    }
  }
  dory_dma_free(&DMA_copy_y);
}
