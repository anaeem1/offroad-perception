// Minimal TensorRT 8.6 engine runner for the two segmentation models. Engines are built on the Jetson
// with trtexec (never shipped from the server). I/O contract (ml/export_onnx.py):
//   input  float32 (1, C, H, W);  probs float32 (1, K, h, w);  labels int32 (1, h, w)
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace offroad {

class TrtEngine {
 public:
  // high_priority: run on the device's highest-priority CUDA stream (the lidar path is the critical path).
  explicit TrtEngine(const std::string& engine_path, bool high_priority = false);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // Copy host input -> device, run, copy outputs back. Blocks until done; returns GPU time in ms.
  float infer(const float* input, std::vector<float>& probs, std::vector<int32_t>& labels);

  const std::vector<int64_t>& input_shape() const { return in_shape_; }
  const std::vector<int64_t>& probs_shape() const { return probs_shape_; }

 private:
  struct Deleter { template <typename T> void operator()(T* p) const { delete p; } };
  std::unique_ptr<nvinfer1::IRuntime, Deleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, Deleter> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, Deleter> context_;
  cudaStream_t stream_{nullptr};
  cudaEvent_t start_{nullptr}, stop_{nullptr};
  void *d_in_{nullptr}, *d_probs_{nullptr}, *d_labels_{nullptr};
  std::vector<int64_t> in_shape_, probs_shape_, labels_shape_;
  size_t in_n_{0}, probs_n_{0}, labels_n_{0};
};

}  // namespace offroad
