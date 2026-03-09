docker run -it --name aideck -v ${PWD}:/module bitcraze/aideck
source /gap_sdk/configs/ai_deck.sh
cd /module/aideck-gap8-examples/examples/other/dory_net_test
make clean all run platform=gvsoc

# docker cp . dd76d00ca0eb:/module/aideck-gap8-examples/examples/other/dory_net_test/src/