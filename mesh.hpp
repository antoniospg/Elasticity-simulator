#ifndef MESH_H
#define MESH_H

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <glad/glad.h>

#include <glm/ext.hpp>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "cuMesh.cuh"
#include "shader.hpp"

struct Vertex {
  glm::vec3 pos;

  Vertex(glm::vec3 pos) : pos(pos) {}
  Vertex() {}
};

class Mesh {
 private:
  unsigned int VBO, VAO, EBO;

 public:
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  cuMesh cm;

  void setupMesh() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex),
                 vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    glBindVertexArray(0);

    setupCuda();
    cm.callKernel();
  }

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
      : vertices(vertices), indices(indices) {
    setupMesh();
  }

  Mesh(const Mesh &m) : vertices(m.vertices), indices(m.indices) {
    setupMesh();
  }

  void render(Shader &shader) {
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
  }

  void setupCuda() {
    cm.mapVBO(VBO);
    cm.mapEBO(EBO);
  }

  ~Mesh() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
  }
};

#endif

