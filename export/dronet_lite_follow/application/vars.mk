FLASH_FILES += hex/ReluConvolution0_weights.hex
FLASH_FILES += hex/ReluConvolution1_weights.hex
FLASH_FILES += hex/Convolution2_weights.hex
FLASH_FILES += hex/Convolution3_weights.hex
FLASH_FILES += hex/ReluConvolution5_weights.hex
FLASH_FILES += hex/ReluConvolution6_weights.hex
FLASH_FILES += hex/Convolution7_weights.hex
FLASH_FILES += hex/Convolution8_weights.hex
FLASH_FILES += hex/ReluConvolution10_weights.hex
FLASH_FILES += hex/ReluConvolution11_weights.hex
FLASH_FILES += hex/ReluConvolution12_weights.hex
FLASH_FILES += hex/FullyConnected14_weights.hex
FLASH_FILES += hex/inputs.hex

READFS_FILES := $(FLASH_FILES)
APP_CFLAGS += -DFS_READ_FS
#PLPBRIDGE_FLAGS += -f