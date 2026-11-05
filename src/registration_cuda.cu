/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

/** @file registration_cuda.cu GPU port of RegistrationImpl::apply.
 *
 * Same algorithm as the CPU version in registration.cpp, split in two kernels:
 *   1. scatter: per depth pixel, undistort, project into color space, and
 *      write its z into a 5x3 window of the min-z "filter map" (atomicMin).
 *   2. gather: per depth pixel, fetch the color pixel, dropping it if a
 *      nearer depth pixel occludes it (filter_tolerance).
 */

#include <libfreenect2/registration_cuda.h>
#include <libfreenect2/logging.h>

#include <cuda_runtime.h>
#include <limits>

#define CHECK_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); return false; } } while(0)
#define CALL_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); } } while(0)

namespace libfreenect2
{

static const int DEPTH_W = 512;
static const int DEPTH_H = 424;
static const int SIZE_DEPTH = DEPTH_W * DEPTH_H;
static const int COLOR_W = 1920;
static const int COLOR_H = 1080;
static const int SIZE_COLOR = COLOR_W * COLOR_H;

struct RegistrationCudaParams
{
  float shift_m;
  float fx;
  float cx;           // color.cx + 0.5f (pre-added for rounding, as on the CPU)
  int fwh;            // filter_width_half
  int fhh;            // filter_height_half
  float tolerance;
  int filter_offset;  // COLOR_W * fhh: border rows above the color image
  int filter_size;    // SIZE_COLOR + 2 * filter_offset
};

__global__ void fillKernel(float *p, int n, float v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = v;
}

__global__ void scatterKernel(const float *__restrict__ depth,
                              const int *__restrict__ map_dist,
                              const float *__restrict__ map_x,
                              const int *__restrict__ map_yi,
                              float *__restrict__ undistorted,
                              int *__restrict__ c_offs,
                              float *filter_map, // NULL if the filter is disabled
                              RegistrationCudaParams p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= SIZE_DEPTH)
    return;

  const int index = map_dist[i];
  if (index < 0)
  {
    c_offs[i] = -1;
    undistorted[i] = 0;
    return;
  }

  const float z = depth[index];
  undistorted[i] = z;

  if (z <= 0.0f)
  {
    c_offs[i] = -1;
    return;
  }

  /* __fdiv_rn: IEEE division even under -use_fast_math, so the truncation to
   * an integer column below matches the CPU implementation. */
  const float rx = (map_x[i] + __fdiv_rn(p.shift_m, z)) * p.fx + p.cx;
  const int cx = rx;
  const int cy = map_yi[i];
  const int c_off = cx + cy * COLOR_W;

  if (c_off < 0 || c_off >= SIZE_COLOR)
  {
    c_offs[i] = -1;
    return;
  }

  c_offs[i] = c_off;

  if (filter_map)
  {
    /* z > 0, and positive IEEE floats order the same as their int bit patterns,
     * so an integer atomicMin is a float min. NaN/inf never win against inf,
     * matching the CPU's `if (z < *it)`. */
    const int zi = __float_as_int(z);
    int yi = p.filter_offset + (cy - p.fhh) * COLOR_W + cx - p.fwh;
    for (int r = -p.fhh; r <= p.fhh; ++r, yi += COLOR_W)
    {
      for (int c = 0; c <= 2 * p.fwh; ++c)
      {
        const int k = yi + c;
        if (k >= 0 && k < p.filter_size)
          atomicMin(reinterpret_cast<int *>(filter_map) + k, zi);
      }
    }
  }
}

__global__ void gatherKernel(const unsigned int *__restrict__ rgb,
                             const float *__restrict__ undistorted,
                             const int *__restrict__ c_offs,
                             const float *__restrict__ filter_map, // NULL if the filter is disabled
                             unsigned int *__restrict__ registered,
                             RegistrationCudaParams p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= SIZE_DEPTH)
    return;

  const int c_off = c_offs[i];
  if (c_off < 0)
  {
    registered[i] = 0;
    return;
  }

  if (filter_map)
  {
    const float min_z = filter_map[p.filter_offset + c_off];
    const float z = undistorted[i];
    registered[i] = __fdiv_rn(z - min_z, z) > p.tolerance ? 0 : rgb[c_off];
  }
  else
  {
    registered[i] = rgb[c_off];
  }
}

class RegistrationCudaImpl
{
public:
  bool good;
  RegistrationCudaParams params;
  cudaStream_t stream;

  int *d_map_dist;
  float *d_map_x;
  int *d_map_yi;

  // Scratch used when the caller's frame is in host memory.
  float *d_depth;
  float *d_undistorted;
  unsigned int *d_registered;
  int *d_c_offs;
  float *d_filter_map;

  RegistrationCudaImpl():
    good(false), stream(0), d_map_dist(NULL), d_map_x(NULL), d_map_yi(NULL),
    d_depth(NULL), d_undistorted(NULL), d_registered(NULL), d_c_offs(NULL), d_filter_map(NULL)
  {
  }

  ~RegistrationCudaImpl()
  {
    if (stream) CALL_CUDA(cudaStreamSynchronize(stream));
    if (d_map_dist) CALL_CUDA(cudaFree(d_map_dist));
    if (d_map_x) CALL_CUDA(cudaFree(d_map_x));
    if (d_map_yi) CALL_CUDA(cudaFree(d_map_yi));
    if (d_depth) CALL_CUDA(cudaFree(d_depth));
    if (d_undistorted) CALL_CUDA(cudaFree(d_undistorted));
    if (d_registered) CALL_CUDA(cudaFree(d_registered));
    if (d_c_offs) CALL_CUDA(cudaFree(d_c_offs));
    if (d_filter_map) CALL_CUDA(cudaFree(d_filter_map));
    if (stream) CALL_CUDA(cudaStreamDestroy(stream));
  }

  bool initialize(const int *distort_map, const float *map_x, const int *map_yi)
  {
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK_CUDA(cudaMalloc(&d_map_dist, SIZE_DEPTH * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_map_x, SIZE_DEPTH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_map_yi, SIZE_DEPTH * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_depth, SIZE_DEPTH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_undistorted, SIZE_DEPTH * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_registered, SIZE_DEPTH * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_c_offs, SIZE_DEPTH * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_filter_map, params.filter_size * sizeof(float)));

    // The lookup tables never change: upload once.
    CHECK_CUDA(cudaMemcpy(d_map_dist, distort_map, SIZE_DEPTH * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_map_x, map_x, SIZE_DEPTH * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_map_yi, map_yi, SIZE_DEPTH * sizeof(int), cudaMemcpyHostToDevice));
    return true;
  }

  bool apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered,
             bool enable_filter, Frame *bigdepth, int *color_depth_map)
  {
    if (!RegistrationCuda::isDevicePointer(rgb->data))
    {
      LOG_ERROR << "RegistrationCuda: color frame is not in device memory";
      return false;
    }
    const unsigned int *rgb_d = reinterpret_cast<const unsigned int *>(rgb->data);

    // Inputs: use device memory in place, upload host memory.
    const float *depth_d = reinterpret_cast<const float *>(depth->data);
    if (!RegistrationCuda::isDevicePointer(depth->data))
    {
      CHECK_CUDA(cudaMemcpyAsync(d_depth, depth->data, SIZE_DEPTH * sizeof(float), cudaMemcpyDefault, stream));
      depth_d = d_depth;
    }

    // Outputs: write device memory in place, otherwise into scratch and download.
    const bool undist_dev = RegistrationCuda::isDevicePointer(undistorted->data);
    const bool reg_dev = RegistrationCuda::isDevicePointer(registered->data);
    const bool coff_dev = color_depth_map && RegistrationCuda::isDevicePointer(color_depth_map);
    const bool big_dev = enable_filter && bigdepth && RegistrationCuda::isDevicePointer(bigdepth->data);

    float *undist_d = undist_dev ? reinterpret_cast<float *>(undistorted->data) : d_undistorted;
    unsigned int *reg_d = reg_dev ? reinterpret_cast<unsigned int *>(registered->data) : d_registered;
    int *coff_d = coff_dev ? color_depth_map : d_c_offs;
    float *filter_d = NULL;
    if (enable_filter)
      filter_d = big_dev ? reinterpret_cast<float *>(bigdepth->data) : d_filter_map;

    const int block = 256;
    if (filter_d)
    {
      fillKernel<<<(params.filter_size + block - 1) / block, block, 0, stream>>>(filter_d, params.filter_size, std::numeric_limits<float>::infinity());
      CHECK_CUDA(cudaGetLastError());
    }

    const int grid = (SIZE_DEPTH + block - 1) / block;
    scatterKernel<<<grid, block, 0, stream>>>(depth_d, d_map_dist, d_map_x, d_map_yi, undist_d, coff_d, filter_d, params);
    CHECK_CUDA(cudaGetLastError());
    gatherKernel<<<grid, block, 0, stream>>>(rgb_d, undist_d, coff_d, filter_d, reg_d, params);
    CHECK_CUDA(cudaGetLastError());

    if (!undist_dev)
      CHECK_CUDA(cudaMemcpyAsync(undistorted->data, undist_d, SIZE_DEPTH * sizeof(float), cudaMemcpyDefault, stream));
    if (!reg_dev)
      CHECK_CUDA(cudaMemcpyAsync(registered->data, reg_d, SIZE_DEPTH * sizeof(unsigned int), cudaMemcpyDefault, stream));
    if (color_depth_map && !coff_dev)
      CHECK_CUDA(cudaMemcpyAsync(color_depth_map, coff_d, SIZE_DEPTH * sizeof(int), cudaMemcpyDefault, stream));
    if (enable_filter && bigdepth && !big_dev)
      CHECK_CUDA(cudaMemcpyAsync(bigdepth->data, filter_d, params.filter_size * sizeof(float), cudaMemcpyDefault, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    return true;
  }
};

RegistrationCuda::RegistrationCuda(const int *distort_map, const float *depth_to_color_map_x, const int *depth_to_color_map_yi,
                                   float color_shift_m, float color_fx, float color_cx,
                                   int filter_width_half, int filter_height_half, float filter_tolerance):
  impl_(new RegistrationCudaImpl())
{
  RegistrationCudaParams &p = impl_->params;
  p.shift_m = color_shift_m;
  p.fx = color_fx;
  p.cx = color_cx;
  p.fwh = filter_width_half;
  p.fhh = filter_height_half;
  p.tolerance = filter_tolerance;
  p.filter_offset = COLOR_W * filter_height_half;
  p.filter_size = SIZE_COLOR + 2 * p.filter_offset;

  impl_->good = impl_->initialize(distort_map, depth_to_color_map_x, depth_to_color_map_yi);
  if (impl_->good)
    LOG_INFO << "CUDA registration initialized";
}

RegistrationCuda::~RegistrationCuda()
{
  delete impl_;
}

bool RegistrationCuda::good() const
{
  return impl_->good;
}

bool RegistrationCuda::apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered,
                             bool enable_filter, Frame *bigdepth, int *color_depth_map)
{
  return impl_->good && impl_->apply(rgb, depth, undistorted, registered, enable_filter, bigdepth, color_depth_map);
}

bool RegistrationCuda::isDevicePointer(const void *ptr)
{
  if (ptr == NULL)
    return false;
  cudaPointerAttributes attr;
  if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess)
  {
    cudaGetLastError(); // clear the sticky-free error so it doesn't leak to later calls
    return false;
  }
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

} /* namespace libfreenect2 */
