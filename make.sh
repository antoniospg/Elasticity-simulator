nvcc -m64 -dc -c -o cuMesh.o cuMesh.cu 
g++ -c -o draw.cpp.o -I/usr/local/cuda/include draw.cpp
g++ -o run -I/usr/local/cuda/include draw.cpp.o cuMesh.o glad.so libassimp.so -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lGL -lGLU -lGLEW -lglfw -lm -ldl
