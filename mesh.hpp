#ifndef MESH_H
#define MESH_H

#include <cuda_runtime.h>
#include <glad/glad.h>
#include <stdio.h>

#include <glm/ext.hpp>
#include <iostream>
#include <vector>

#include "cuMesh.cuh"
#include "shader.hpp"

using namespace std;

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;

  Vertex(glm::vec3 pos) : pos(pos) {}
  Vertex(float x, float y, float z) : pos(x, y, z) {}
  Vertex() {}
};

class Mesh {
 private:
  unsigned int VBO, VAO, EBO;

 public:
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  size_t n_vertices, n_tris;
  std::vector<unsigned int> indices3;
  cuMesh cm;
  bool empty_mesh;

  void setupMesh() {
    n_vertices = vertices.size();
    n_tris = indices.size() / 3;
    cm = cuMesh((vert3 *)vertices.data(), (int3 *)indices.data(),
                vertices.size(), indices.size() / 3, false);

    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, normal));
    glBindVertexArray(0);
  }

  Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
      : vertices(vertices), indices(indices) {
    setupMesh();
    empty_mesh = false;
  }

  Mesh(vert3 *d_vertices_in, int3 *d_indices_in, size_t n_vertices,
       size_t n_indices, bool device_pointers) {
    n_vertices = n_vertices;
    n_tris = n_indices;

    cm = cuMesh(d_vertices_in, d_indices_in, n_vertices, n_indices,
                device_pointers);

    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, normal));
    glBindVertexArray(0);
  }

  Mesh(const Mesh &m) : vertices(m.vertices), indices(m.indices) {
    setupMesh();
    empty_mesh = false;
  }

  Mesh() { empty_mesh = true; }

  void render(Shader &shader) {
    glBindVertexArray(cm.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, cm.VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cm.EBO);

    glDrawElements(GL_TRIANGLES, n_tris * 3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  }

  ~Mesh() {
    if (!empty_mesh) cm.deleteVBO_CUDA();
  }
};

#endif
