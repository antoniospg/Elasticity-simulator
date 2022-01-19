#ifndef MODEL_H
#define MODEL_H

#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <assimp/Importer.hpp>
#include <glm/ext.hpp>
#include <string>
#include <vector>

#include "mesh.hpp"
#include "shader.hpp"

class Model {
 public:
  Mesh processMesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    for (size_t i = 0; i < mesh->mNumVertices; i++) {
      // vertex info
      glm::vec3 vector;
      vector.x = mesh->mVertices[i].x;
      vector.y = mesh->mVertices[i].y;
      vector.z = mesh->mVertices[i].z;

      Vertex vertex(vector);
      vertices.push_back(vertex);
    }

    // index info
    for (size_t i = 0; i < mesh->mNumFaces; i++) {
      aiFace face = mesh->mFaces[i];
      for (size_t j = 0; j < face.mNumIndices; j++)
        indices.push_back(face.mIndices[j]);
    }

    return Mesh(vertices, indices);
  }

  void processNode(aiNode* node, const aiScene* scene) {
    for (size_t i = 0; i < node->mNumMeshes; i++) {
      aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
      meshes.push_back(processMesh(mesh, scene));
    }

    for (size_t i = 0; i < node->mNumChildren; i++) {
      processNode(node->mChildren[i], scene);
    }
  }

  void loadModel(std::string path) {
    Assimp::Importer imp;
    const aiScene* scene = imp.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
        !scene->mRootNode) {
      std::cout << "ERROR::ASSIMP::" << imp.GetErrorString() << std::endl;
      return;
    }
    directory = path.substr(0, path.find_last_of('/'));

    processNode(scene->mRootNode, scene);
  }

  Model(std::string path) { loadModel(path); }

  void render(Shader& shader) {
    for (auto& m : meshes) {
      m.render(shader);
    }
  }

 private:
  std::vector<Mesh> meshes;
  std::string directory;
};

#endif

