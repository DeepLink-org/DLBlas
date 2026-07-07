maca_cc=$(MACA_PATH)tools/cu-bridge/bin/cucc
include= -I $(MACA_PATH)tools/cu-bridge/include/ -I ./inc/
LIBS=-lmcruntime -lmxc-runtime64 -lcurand -lmcToolsExt
CFLAG=-D_MACACC_RTC__ -D__FAST_HALF_CVT__ -O3 -arch=sm_80 -use-dim-intrinsic -mllvm -metaxgpu-new-streg-abi=true --fast-bfloat16-math rz -DUSE_MACA
objs=./src/tmp_test.cu
test_maca:$(objs)
	$(maca_cc) $(CFLAG) $(include) $^ -o test_maca $(LIBS) --kernel-probe 0
clean:
	$(RM) *.o *.asm *.s *.bc *.macai *.out *.pftrace *.mcfb *.dat_dpc_0 *.json test_maca test -r
