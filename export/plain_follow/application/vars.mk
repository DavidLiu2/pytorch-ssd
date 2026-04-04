FLASH_FILES += hex/ReluConvolution0_weights.hex
FLASH_FILES += hex/ReluConvolution1_weights.hex
FLASH_FILES += hex/ReluConvolution2_weights.hex
FLASH_FILES += hex/ReluConvolution3_weights.hex
FLASH_FILES += hex/ReluConvolution4_weights.hex
FLASH_FILES += hex/ReluConvolution5_weights.hex
FLASH_FILES += hex/ReluConvolution6_weights.hex
FLASH_FILES += hex/FullyConnected8_weights.hex
FLASH_FILES += hex/inputs.hex

READFS_FILES := $(FLASH_FILES)
APP_CFLAGS += -DFS_READ_FS
#PLPBRIDGE_FLAGS += -f