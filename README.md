# STEM Networking Benchmark

This application is a high-performance benchmarking tool designed for EELS Microscopy data acquisition and processing pipelines. It makes use of the [NVIDIA Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk) to implement a modular, GPU-accelerated pipeline that receives high-speed UDP network packets, aggregates them into frames, processes them using PyTorch, and writes the results to disk.

## Architecture & Strategy

The application demonstrates a **modular, operator-based architecture** where distinct functional blocks are encapsulated as Holoscan Operators.

### Pipeline Overview

```mermaid
graph LR
    A[Network Source / HDF5 Source] -->|UDP Packets / Raw Data| B(StemReceiverOp / HDF5ReplayerOp)
    B -->|holoscan::TensorMap (GPU)| C(PyTorchProcessorOp)
    C -->|holoscan::TensorMap (GPU)| D(HDF5WriterOp)
    D -->|HDF5 File| E[Disk]
```

## Operators

### 1. `StemReceiverOp`
    *   Interfaces with the Holoscan Advanced Network operator (using DPDK).
    *   Aggregates incoming UDP packets into full 2D frames.
    *   Uses a custom CUDA kernel (`gather_packets`) to handle packet reordering and memory alignment, ensuring robust handling of arbitrary packet sizes.
    *   Emits a `holoscan::TensorMap` containing the raw frame data on the GPU.

### 2. `HDF5ReplayerOp`
    *   Acts as a file-based source for testing and benchmarking without live network hardware.
    *   Reads pre-recorded frames from an HDF5 file.
    *   Uploads frame data to GPU memory.
    *   Emits a `holoscan::TensorMap` identical to the receiver's output.

### 3. `PyTorchProcessorOp`
    *   Receives the GPU tensor from the receiver.
    *   Wraps the memory in a `torch::Tensor`.
    *   Performs processing in PyTorch.
    *   Emits the processed result as a new tensor.

### 4. `HDF5WriterOp`
    *   Receives the processed tensor.
    *   Transfers data from GPU to Host memory.
    *   Writes the frame to an HDF5 file for offline analysis and verification.

## Acknowledgements

This project is built on the NVIDIA holoscan SDK and holohub.

- [NVIDIA Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk)
- [NVIDIA HoloHub](https://github.com/nvidia-holoscan/holohub)
