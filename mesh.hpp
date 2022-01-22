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
  glm::vec3 color = {0.0f, 0.0f, 0.0f};

  Vertex(glm::vec3 pos) : pos(pos) {}
  Vertex() {}
};

class Chunks {
 public:
  array<vector<int2>, 4> colors;
  int2 *links_h;
  int n_l;

  Chunks(int3 *indices_h, size_t n_i, float3 *vertices_h, size_t n_v,
         size_t block_size, std::vector<Vertex> &vertices) {
    // Inicializar links_h
    std::vector<int2> links_h;
    n_l = 3 * n_i;

    // Calcular links e map de vertice em link
    map<int, vector<int>> get_links;
    map<int, int> get_next_vertex;

    // Less function for int2
    auto comp_int2 = [](int2 a, int2 b) {
      return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
    };
    set<int2, decltype(comp_int2)> links_visited(comp_int2);

    for (size_t i = 0; i < n_i; i++) {
      // Check if halfedge or opposite halfedge were already visited
      int2 hf;

      hf = {indices_h[i].x, indices_h[i].y};
      if (links_visited.find(hf) == links_visited.end() &&
          links_visited.find(int2{hf.y, hf.x}) == links_visited.end()) {
        links_h.push_back(hf);
        get_links[hf.x].push_back(links_h.size() - 1);
        get_links[hf.y].push_back(links_h.size() - 1);
        get_next_vertex[links_h.size() - 1] = hf.y;

        links_visited.insert(hf);
        links_visited.insert(int2{hf.y, hf.x});
      }

      hf = {indices_h[i].y, indices_h[i].z};
      if (links_visited.find(hf) == links_visited.end() &&
          links_visited.find(int2{hf.y, hf.x}) == links_visited.end()) {
        links_h.push_back(hf);
        get_links[hf.x].push_back(links_h.size() - 1);
        get_links[hf.y].push_back(links_h.size() - 1);
        get_next_vertex[links_h.size() - 1] = hf.y;

        links_visited.insert(hf);
        links_visited.insert(int2{hf.y, hf.x});
      }

      hf = {indices_h[i].z, indices_h[i].x};
      if (links_visited.find(hf) == links_visited.end() &&
          links_visited.find(int2{hf.y, hf.x}) == links_visited.end()) {
        links_h.push_back(hf);
        get_links[hf.x].push_back(links_h.size() - 1);
        get_links[hf.y].push_back(links_h.size() - 1);
        get_next_vertex[links_h.size() - 1] = hf.y;

        links_visited.insert(hf);
        links_visited.insert(int2{hf.y, hf.x});
      }
    }

    // Colorir malha
    queue<int> q_vertex;
    set<int> vertices_visited;
    q_vertex.push(0);
    links_visited.clear();
    int first_color = 0;     // 0..3
    int second_color = 0;    // 0..5
    size_t count_links = 0;  // max = block_size

    // Debugar cores visualmente-> 0=azul, 1=vermelho, 2=verde, 3=amarelo,
    // nada=preto
    std::vector<glm::vec3> visual_colors = {{0.0f, 0.0f, 1.0f},
                                            {1.0f, 0.0f, 0.0f},
                                            {0.0f, 1.0f, 0.0f},
                                            {1.0f, 1.0f, 0.0f}};

    int land = 0;
    while (!q_vertex.empty()) {
      int curr = q_vertex.front();
      q_vertex.pop();
      if (vertices_visited.find(curr) != vertices_visited.end()) continue;
      vertices_visited.insert(curr);

      for (auto link : get_links[curr]) {
        if (links_visited.find(links_h[link]) != links_visited.end() || links_visited.find(int2{links_h[link].y, links_h[link].x}) != links_visited.end()) continue;
        links_visited.insert(links_h[link]);

        count_links++;
        colors[first_color].push_back(links_h[link]);
        q_vertex.push(get_next_vertex[link]);

        // Debug : atribuir cor nos vertices
        if (first_color == 1) {
          vertices[links_h[link].x].color = visual_colors[land];
          vertices[links_h[link].y].color = visual_colors[land];

          std::cout << links_h[link].x << std::endl;
        }

        if (count_links == block_size) {
          first_color = (first_color + 1) % 4;
          if (first_color == 1) land = (land + 1) % 4;
          count_links = 0;
        }
      }
    }
  }
};

class Mesh {
 private:
  unsigned int VBO, VAO, EBO;

 public:
  std::vector<Vertex> vertices;
  std::vector<float3> vertices_h;
  std::vector<unsigned int> indices;
  cuMesh cm;

  void cpVertexPos() {
    vertices_h.clear();
    vertices_h.reserve(vertices.size());

    for (auto v : vertices) {
      float3 v_h;
      v_h.x = v.pos.x;
      v_h.y = v.pos.y;
      v_h.z = v.pos.z;

      vertices_h.push_back(v_h);
    }
  }

  void setupMesh() {
    cpVertexPos();

    Chunks ch((int3 *)indices.data(), indices.size() / 3, vertices_h.data(),
              vertices_h.size(), 2, vertices);

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
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void *)offsetof(Vertex, color));
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

