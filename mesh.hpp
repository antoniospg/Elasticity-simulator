#ifndef MESH_H
#define MESH_H

#include "shader.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/ext.hpp>
#include <cuda_runtime.h>
#include "cuMesh.cuh"

#include <vector>

struct Vertex {
  glm::vec3 pos;

  Vertex(glm::vec3 pos): pos(pos) {}
  Vertex(){}
};

class Mesh {
private:
  unsigned int VBO, VAO, EBO;
  struct cudaGraphicsResource* positionsVBO_CUDA;

public:
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;

  void setupMesh() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
  
    glBindVertexArray(VAO);
  
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glBindVertexArray(0);
  }

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices):
    vertices(vertices), indices(indices) {setupMesh();}

  Mesh(const Mesh& m):
    vertices(m.vertices), indices(m.indices) {setupMesh();}

  void render(Shader& shader) {
    setupCuda();
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void setupCuda() {
    cuMesh::mapVBO(VBO);
    cuMesh::callKernel();
    cuMesh::deleteVBO_CUDA();
  }

  ~Mesh(){
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
  }
};

#endif

