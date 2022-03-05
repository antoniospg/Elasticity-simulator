#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <glm/ext.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ioUtils.hpp"
#include "mesh.hpp"
#include "model.hpp"
#include "shader.hpp"
#include "voxelModel.hpp"

int main() {
  float h = 1200;
  const char* glsl_version = "#version 130";

  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // glfw window creation
  // --------------------
  GLFWwindow* window = glfwCreateWindow(h, h, "-----", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, ioUtils::framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  VoxelModel vm("foot.raw");
  int isoVal = 0;
  Shader df("shaders/default.vert", "shaders/default.frag");
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glPolygonMode(GL_FRONT, GL_POLYGON);

  // render loop
  // -----------
  double xpos = h / 2, ypos = h / 2;
  double xpos0 = xpos, ypos0 = ypos;
  bool leftClick = false;
  bool rightClick = false;
  glm::vec3 camPos(0.0f, 0.0f, 1400.0f);
  float timeVal = glfwGetTime();

  glm::mat4 proj = glm::ortho(-h / 2, h / 2, -h / 2, h / 2, 0.0f, 10 * 1200.0f);
  glm::mat4 view = glm::translate(glm::mat4(1.0f), -camPos);
  glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(3000.0f));

  float scale = 1.0f;
  while (!glfwWindowShouldClose(window)) {
    // input
    // -----
    leftClick = false;
    rightClick = false;
    glm::vec3 camPos0 = camPos;
    ioUtils::processInput(window, leftClick, rightClick, camPos);
    view = glm::translate(view, camPos0 - camPos);

    // render
    // ------
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Count fps
    float prevTime = timeVal;
    timeVal = glfwGetTime();
    string title = to_string(1 / (timeVal - prevTime));
    glfwSetWindowTitle(window, title.data());

    // feed inputs to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // render your GUI
    ImGui::Begin("Set isoVal");
    ImGui::SliderInt("isoVal", &isoVal, 1, 1000);
    ImGui::End();

    // Render dear imgui into screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // update shader uniform
    glfwGetCursorPos(window, &xpos, &ypos);
    if (leftClick) {
      float u1 = (xpos - xpos0) * 0.1f;
      float u2 = (ypos - ypos0) * 0.1f;
      glm::mat4 y_rot = glm::rotate(glm::mat4(1.0f), glm::radians(u2),
                                    glm::vec3(1.0f, 0.0f, 0.0f));
      glm::mat4 x_rot = glm::rotate(glm::mat4(1.0f), glm::radians(u1),
                                    glm::vec3(0.0f, 1.0f, 0.0f));
      model = y_rot * x_rot * model;
    } else if (rightClick) {
      float u1 = (ypos - ypos0) * 0.01f;
      u1 = std::max(0.0f, 1.0f + u1);
      view = glm::scale(view, glm::vec3(u1, u1, u1));
    }
    xpos0 = xpos, ypos0 = ypos;

    df.compile();
    df.setShader();
    df.setMat4("proj", proj);
    df.setMat4("view", view);
    df.setMat4("model", model);

    // Model m("./second.obj");

    // m.render(df);
    vm.draw(df, isoVal);

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
    // etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  return 0;
}
