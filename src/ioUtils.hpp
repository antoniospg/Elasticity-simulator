#ifndef IOUTILS_H
#define IOUTILS_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/ext.hpp>

namespace ioUtils{

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window, bool& leftClick, bool& rightClick, glm::vec3& camPos) {
  const float camSpeed = 1.5f;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    leftClick = true;
  if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    rightClick = true;
  if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    camPos.y += camSpeed;
  if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    camPos.y -= camSpeed;
  if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    camPos.x -= camSpeed;
  if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    camPos.x += camSpeed;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width and 
  // height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

}
#endif

