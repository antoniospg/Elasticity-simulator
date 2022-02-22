#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 vertColor;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

void main() {
  vertColor = vec3(1.0f, 1.0f, 1.0f);
  gl_Position = proj*view*model*vec4(aPos, 1.0f);
}
