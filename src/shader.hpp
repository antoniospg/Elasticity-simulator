#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/ext.hpp>

class Shader {
private:
  std::string vertPath;
  std::string fragPath;

  //Aux function
  std::string readFile  (const char *filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);
  
    if(!fileStream.is_open()) {
      std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
      return "";
    }
  
    std::string line = "";
    while(!fileStream.eof()) {
      std::getline(fileStream, line);
      content.append(line + "\n");
    }
  
    fileStream.close();
    return content;
  }

  void checkErrors (GLuint shader, GLenum pname) {
    int success;
    char infoLog[512];

    if (pname == GL_COMPILE_STATUS) {
      glGetShaderiv(shader, pname, &success);

      if (!success) {
        glGetProgramInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER :\n" << infoLog << "\n################" <<  std::endl;
      }
    }
    else if (pname == GL_LINK_STATUS) {
      glGetProgramiv(shader, pname, &success);

      if (!success) {
        glGetProgramInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER :\n" << infoLog << "\n################" <<  std::endl;
      }
    }
  }

public:
  int ID;

  Shader (std::string vertPath, std::string fragPath): vertPath(vertPath), fragPath(fragPath) {} 

  void setMat4(const std::string& name, const glm::mat4& mat) {
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
  }

  void setFloat(const std::string& name, float value) {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value); 
  }

  void compile () {
    std::string vertString = readFile(vertPath.c_str());
    std::string fragString = readFile(fragPath.c_str());
    const char* vertSource = vertString.c_str(); 
    const char* fragSource = fragString.c_str(); 

    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertSource, NULL);
    glCompileShader(vertexShader);

    checkErrors(vertexShader, GL_COMPILE_STATUS);

    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragSource, NULL);
    glCompileShader(fragmentShader);

    checkErrors(fragmentShader, GL_COMPILE_STATUS);

    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    ID = shaderProgram;
    glAttachShader(ID, vertexShader);
    glAttachShader(ID, fragmentShader);
    glLinkProgram(ID);

    // check for linking errors
    checkErrors(ID, GL_LINK_STATUS);

    glDeleteShader(vertexShader);                         
    glDeleteShader(fragmentShader);
  }

  void setShader() {
    // be sure to activate the shader before any calls to glUniform
    glUseProgram(ID);
  }

  void disableShader() {
    glUseProgram(0);
  }

  ~Shader () {
    glDeleteProgram(ID);
  }
};

#endif

