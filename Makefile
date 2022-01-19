# Product Names
CUDA_OBJ = cuda.o

# Input Names
CUDA_FILES = cuMesh.cu
CPP_FILES = draw.cpp

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -g -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS = 


# CUDA Object Files
CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -g -Wall -D_REENTRANT -std=c++0x -pthread
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart glad.so libassimp.so -lGL -lGLU -lGLEW -lglfw -lm -ldl

# ------------------------------------------------------------------------------
# Make Rules (Lab 3 specific)
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ_RUN = $(addprefix run-, $(notdir $(addsuffix .o, $(CPP_FILES))))

# Top level rules
all: run

run: $(OBJ_RUN) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o run $(INCLUDE) $^ $(LIBS) 

# Compile C++ Source Files
run-%.cpp.o: %.cpp 
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $< 

# Compile CUDA Source Files
%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

$(CUDA_OBJ): $(CUDA_OBJ_FILES)
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f *.o
	rm -f *.o

.PHONY: clean
