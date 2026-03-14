#include "mem.h"
#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"

#ifdef USE_HYPERFLASH
#include "bsp/flash/hyperflash.h"
typedef struct pi_hyperflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_hyperflash_conf_init(conf)
#elif defined USE_SPIFLASH
#include "bsp/flash/spiflash.h"
typedef struct pi_spiflash_conf flash_conf_t;
#define flash_conf_init(conf) pi_spiflash_conf_init(conf)
#elif defined USE_MRAM
typedef struct pi_mram_conf flash_conf_t;
#define flash_conf_init(conf) pi_mram_conf_init(conf)
#else
typedef struct pi_default_flash_conf flash_conf_t;
#define flash_conf_init(conf) pi_default_flash_conf_init(conf)
#endif

#ifdef USE_HYPERRAM
#include "bsp/ram/hyperram.h"
typedef struct pi_hyperram_conf ram_conf_t;
#define ram_conf_init(conf) pi_hyperram_conf_init(conf)
#else
typedef struct pi_default_ram_conf ram_conf_t;
#define ram_conf_init(conf) pi_default_ram_conf_init(conf)
#endif

#define BUFFER_SIZE 128
static uint8_t buffer[BUFFER_SIZE];

static struct pi_device flash;
static flash_conf_t flash_conf;

static struct pi_device fs;
static struct pi_readfs_conf fs_conf;

static struct pi_device ram;
static ram_conf_t ram_conf;

static int hostfs_open_write(struct pi_device *host_fs, pi_fs_file_t **fd, const char *filename) {
  struct pi_fs_conf conf;

  pi_fs_conf_init(&conf);
  conf.type = PI_FS_HOST;
  pi_open_from_conf(host_fs, &conf);

  if (pi_fs_mount(host_fs)) {
    printf("ERROR: Cannot mount hostfs for %s\n", filename);
    return -1;
  }

  *fd = pi_fs_open(host_fs, filename, PI_FS_FLAGS_WRITE);
  if (*fd == NULL) {
    printf("ERROR: Cannot open hostfs file %s for write\n", filename);
    pi_fs_unmount(host_fs);
    return -2;
  }

  return 0;
}


void mem_init() {
  flash_conf_init(&flash_conf);
  pi_open_from_conf(&flash, &flash_conf);
  if (pi_flash_open(&flash)) {
    printf("ERROR: Cannot open flash! Exiting...\n");
    pmsis_exit(-1);
  }

  /* Open filesystem on flash. */
  pi_readfs_conf_init(&fs_conf);
  fs_conf.fs.flash = &flash;
  pi_open_from_conf(&fs, &fs_conf);
  if (pi_fs_mount(&fs)) {
    printf("ERROR: Cannot mount filesystem! Exiting...\n");
    pmsis_exit(-2);
  }

  ram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  if (pi_ram_open(&ram)) {
    printf("ERROR: Cannot open ram! Exiting...\n");
    pmsis_exit(-3);
  }
}

struct pi_device *get_ram_ptr() {
 return &ram;
}

void *ram_malloc(size_t size) {
  void *ptr = NULL;
  pi_ram_alloc(&ram, &ptr, size);
  return ptr;
}

void ram_free(void *ptr, size_t size) {
  pi_ram_free(&ram, ptr, size);
}

void ram_read(void *dest, void *src, const size_t size) {
  pi_ram_read(&ram, src, dest, size);
}

void ram_write(void *dest, void *src, const size_t size) {
  pi_ram_write(&ram, dest, src, size);
}

void *cl_ram_malloc(size_t size) {
  int addr;
  pi_cl_ram_req_t req;
  pi_cl_ram_alloc(&ram, size, &req);
  pi_cl_ram_alloc_wait(&req, &addr);
  return (void *) addr;
}

void cl_ram_free(void *ptr, size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_free(&ram, ptr, size, &req);
  pi_cl_ram_free_wait(&req);
}

void cl_ram_read(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_read(&ram, src, dest, size, &req);
  pi_cl_ram_read_wait(&req);
}

void cl_ram_write(void *dest, void *src, const size_t size) {
  pi_cl_ram_req_t req;
  pi_cl_ram_write(&ram, dest, src, size, &req);
  pi_cl_ram_write_wait(&req);
}

size_t load_file_to_ram(const void *dest, const char *filename) {
  pi_fs_file_t *fd = pi_fs_open(&fs, filename, 0);
  if (fd == NULL) {
    printf("ERROR: Cannot open file %s! Exiting...", filename);
    pmsis_exit(-4);
  }

  const size_t size = fd->size;

  size_t offset = 0;
  do {
    const size_t read_bytes = pi_fs_read(fd, buffer, BUFFER_SIZE);
    ram_write(dest + offset, buffer, read_bytes);
    offset += read_bytes;
  } while (offset < size);

  return offset;
}

int hostfs_write_file(const char *filename, const void *src, size_t size) {
  struct pi_device host_fs;
  pi_fs_file_t *fd = NULL;
  const uint8_t *src_bytes = (const uint8_t *) src;
  size_t offset = 0;
  int ret = 0;

  ret = hostfs_open_write(&host_fs, &fd, filename);
  if (ret != 0) {
    return ret;
  }

  while (offset < size) {
    size_t chunk = BUFFER_SIZE;
    if (chunk > (size - offset)) {
      chunk = size - offset;
    }

    if ((size_t) pi_fs_write(fd, (void *) (src_bytes + offset), chunk) != chunk) {
      printf("ERROR: hostfs write failed for %s at offset %u\n", filename, (unsigned int) offset);
      ret = -3;
      break;
    }

    offset += chunk;
  }

  pi_fs_close(fd);
  pi_fs_unmount(&host_fs);
  return ret == 0 ? (int) offset : ret;
}

int hostfs_write_file_from_ram(const char *filename, void *ram_src, size_t size) {
  struct pi_device host_fs;
  pi_fs_file_t *fd = NULL;
  size_t offset = 0;
  int ret = 0;

  ret = hostfs_open_write(&host_fs, &fd, filename);
  if (ret != 0) {
    return ret;
  }

  while (offset < size) {
    size_t chunk = BUFFER_SIZE;
    if (chunk > (size - offset)) {
      chunk = size - offset;
    }

    ram_read(buffer, (uint8_t *) ram_src + offset, chunk);
    if ((size_t) pi_fs_write(fd, buffer, chunk) != chunk) {
      printf("ERROR: hostfs RAM write failed for %s at offset %u\n", filename, (unsigned int) offset);
      ret = -3;
      break;
    }

    offset += chunk;
  }

  pi_fs_close(fd);
  pi_fs_unmount(&host_fs);
  return ret == 0 ? (int) offset : ret;
}
