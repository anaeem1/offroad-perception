// Minimal .npy reader/writer (little-endian, C order, version 1.0) for parity tests against the Python
// reference pipeline. Supports the dtypes the tests exchange: f4, f8, i4, i8, u1, u4.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace offroad::npy {

template <typename T> const char* descr();
template <> inline const char* descr<float>() { return "<f4"; }
template <> inline const char* descr<double>() { return "<f8"; }
template <> inline const char* descr<int32_t>() { return "<i4"; }
template <> inline const char* descr<int64_t>() { return "<i8"; }
template <> inline const char* descr<uint8_t>() { return "|u1"; }
template <> inline const char* descr<uint32_t>() { return "<u4"; }

template <typename T>
struct Array {
  std::vector<size_t> shape;
  std::vector<T> data;
  size_t size() const { return data.size(); }
};

template <typename T>
Array<T> load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("not npy: " + path);
  uint8_t ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) { uint16_t h; f.read(reinterpret_cast<char*>(&h), 2); hlen = h; }
  else { f.read(reinterpret_cast<char*>(&hlen), 4); }
  std::string header(hlen, ' ');
  f.read(header.data(), hlen);
  if (header.find(descr<T>()) == std::string::npos) throw std::runtime_error("dtype mismatch in " + path + ": " + header);
  if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error("fortran order: " + path);
  Array<T> a;
  auto p = header.find('(') + 1, q = header.find(')');
  std::string dims = header.substr(p, q - p);
  size_t n = 1, pos = 0;
  while (pos < dims.size()) {
    auto comma = dims.find(',', pos);
    std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (tok.find_first_not_of(' ') != std::string::npos) { a.shape.push_back(std::stoul(tok)); n *= a.shape.back(); }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  a.data.resize(n);
  f.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (!f) throw std::runtime_error("short read: " + path);
  return a;
}

template <typename T>
void save(const std::string& path, const std::vector<size_t>& shape, const T* data) {
  std::string dims;
  size_t n = 1;
  for (size_t d : shape) { dims += std::to_string(d) + ","; n *= d; }
  if (shape.size() > 1) dims.pop_back();
  std::string header = std::string("{'descr': '") + descr<T>() + "', 'fortran_order': False, 'shape': (" + dims + "), }";
  size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t hlen = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&hlen), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(T)));
}

}  // namespace offroad::npy
