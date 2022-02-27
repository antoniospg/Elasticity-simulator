  if (device_pointers) {
    cudaMemcpy(d_vertices, vertices_in, n_vertices * sizeof(float3),
               cudaMemcpyDeviceToDevice);

