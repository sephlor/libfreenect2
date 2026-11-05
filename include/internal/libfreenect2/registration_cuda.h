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

/** @file registration_cuda.h GPU implementation of Registration::apply. */

#ifndef REGISTRATION_CUDA_H_
#define REGISTRATION_CUDA_H_

#include <libfreenect2/frame_listener.hpp>

namespace libfreenect2
{

class RegistrationCudaImpl;

/** Runs Registration::apply on the GPU. Used when the color frame lives in
 * CUDA device memory (e.g. from NvJpegRgbPacketProcessor).
 *
 * Any frame argument may be in host or device memory: host inputs are
 * uploaded, host outputs are downloaded, device pointers are used in place.
 * The call is synchronous: outputs are complete when it returns. */
class RegistrationCuda
{
public:
  RegistrationCuda(const int *distort_map, const float *depth_to_color_map_x, const int *depth_to_color_map_yi,
                   float color_shift_m, float color_fx, float color_cx,
                   int filter_width_half, int filter_height_half, float filter_tolerance);
  ~RegistrationCuda();

  bool good() const;

  bool apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered,
             bool enable_filter, Frame *bigdepth, int *color_depth_map);

  /** True if ptr is CUDA device (or managed) memory. */
  static bool isDevicePointer(const void *ptr);

private:
  RegistrationCudaImpl *impl_;
};

} /* namespace libfreenect2 */
#endif /* REGISTRATION_CUDA_H_ */
