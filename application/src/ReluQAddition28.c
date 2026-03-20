#include "ReluQAddition28.h"
#include "pulp.h"
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"

void ReluQAddition28(void *args) {
  unsigned int *real_arg = (unsigned int *) args;
  const int32_t *x = (const int32_t *) real_arg[3];
  const uint8_t *x2 = (const uint8_t *) real_arg[4];
  uint8_t *y = (uint8_t *) real_arg[5];
  unsigned int out_mult_in = (unsigned int) real_arg[9];
  unsigned int out_shift_in = (unsigned int) real_arg[10];

  pulp_nn_add_raw_i32_u8_mixed(x, x2, y, 32, 32, 6, out_mult_in, out_shift_in, 1280u, "ReluQAddition28");
}
