#ifndef MESH_H
#define MESH_H

#include <cuda_runtime.h>
#include <glad/glad.h>

#include <glm/ext.hpp>
#include <iostream>
#include <vector>

#include "cuMesh.cuh"
#include "shader.hpp"

using namespace std;

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
    cm = cuMesh((float3 *)vertices.data(), (uint3 *)indices.data(),
                vertices.size(), indices.size() / 3);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    glBindVertexArray(0);
  }

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
      : vertices(vertices), indices(indices) {
    setupMesh();
  }

  Mesh(const Mesh &m) : vertices(m.vertices), indices(m.indices) {
    setupMesh();
  }

  Mesh(int n_vertices, int n_indices) {
    cm = cuMesh(n_vertices, n_indices);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    glBindVertexArray(0);
  }

  void render(Shader &shader) {
    glBindVertexArray(cm.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, cm.VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cm.EBO);

    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }

  ~Mesh() { cm.deleteVBO_CUDA(); }
};

#endif
