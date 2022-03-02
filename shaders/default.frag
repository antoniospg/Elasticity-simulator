#version 330 core
in vec3 vn;
out vec4 FragColor;
void main() {
  vec3 pos = vec3(0.0, 1.0, 0.0);
  float val = dot(pos, vn)/2;
  val = 1.0;
  FragColor = vec4(val, val, val, 1.0);
}
