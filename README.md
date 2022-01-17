# Interactive CUDA Renderer

Controls:
- WASD for camera movement
- Arrow keys for camera direction
- ESC to close

Tested With:
- Linux
- NVIDIA 495 Drivers
- CUDA Toolkit 11.5
- OpenGL 4.6
- GLFW 3.3
- GLEW 2.1
- Boost Fusion 1.78

Building:
```
git clone --recursive https://github.com/matthew-haines/cuda-renderer.git
cd cuda-renderer && mkdir build && cmake . -B build && cmake --build build
build/rt
```

Performance:
~250 MRay/s on a GTX 970. (~30 fps at 1200x900 with 8 rays/pixel)
