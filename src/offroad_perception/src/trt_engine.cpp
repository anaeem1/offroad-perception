#include "offroad_perception/trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace offroad {
namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << "\n";
  }
};
Logger g_logger;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::vector<int64_t> dims_of(const nvinfer1::Dims& d) { return std::vector<int64_t>(d.d, d.d + d.nbDims); }
size_t count(const std::vector<int64_t>& s) { return std::accumulate(s.begin(), s.end(), size_t{1}, std::multiplies<size_t>()); }

}  // namespace

TrtEngine::TrtEngine(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open engine " + path);
  std::vector<char> blob(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(blob.data(), static_cast<std::streamsize>(blob.size()));
  runtime_.reset(nvinfer1::createInferRuntime(g_logger));
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("cannot deserialize " + path);
  context_.reset(engine_->createExecutionContext());
  in_shape_ = dims_of(engine_->getTensorShape("input"));
  probs_shape_ = dims_of(engine_->getTensorShape("probs"));
  labels_shape_ = dims_of(engine_->getTensorShape("labels"));
  in_n_ = count(in_shape_);
  probs_n_ = count(probs_shape_);
  labels_n_ = count(labels_shape_);
  check(cudaStreamCreate(&stream_), "stream");
  check(cudaEventCreate(&start_), "event");
  check(cudaEventCreate(&stop_), "event");
  check(cudaMalloc(&d_in_, in_n_ * sizeof(float)), "malloc input");
  check(cudaMalloc(&d_probs_, probs_n_ * sizeof(float)), "malloc probs");
  check(cudaMalloc(&d_labels_, labels_n_ * sizeof(int32_t)), "malloc labels");
  context_->setTensorAddress("input", d_in_);
  context_->setTensorAddress("probs", d_probs_);
  context_->setTensorAddress("labels", d_labels_);
}

TrtEngine::~TrtEngine() {
  cudaFree(d_in_);
  cudaFree(d_probs_);
  cudaFree(d_labels_);
  if (start_) cudaEventDestroy(start_);
  if (stop_) cudaEventDestroy(stop_);
  if (stream_) cudaStreamDestroy(stream_);
}

float TrtEngine::infer(const float* input, std::vector<float>& probs, std::vector<int32_t>& labels) {
  probs.resize(probs_n_);
  labels.resize(labels_n_);
  check(cudaMemcpyAsync(d_in_, input, in_n_ * sizeof(float), cudaMemcpyHostToDevice, stream_), "h2d");
  check(cudaEventRecord(start_, stream_), "record");
  if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
  check(cudaEventRecord(stop_, stream_), "record");
  check(cudaMemcpyAsync(probs.data(), d_probs_, probs_n_ * sizeof(float), cudaMemcpyDeviceToHost, stream_), "d2h");
  check(cudaMemcpyAsync(labels.data(), d_labels_, labels_n_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_), "d2h");
  check(cudaStreamSynchronize(stream_), "sync");
  float ms = 0;
  cudaEventElapsedTime(&ms, start_, stop_);
  return ms;
}

}  // namespace offroad
