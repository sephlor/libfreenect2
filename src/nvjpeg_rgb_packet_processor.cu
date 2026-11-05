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

/** @file nvjpeg_rgb_packet_processor.cu JPEG decoder using CUDA nvJPEG.
 *
 * Follows the same data flow as CudaDepthPacketProcessor:
 *   USB -> pinned host input buffer (our allocator) -> device decode
 *   -> device BGRX -> one D2H copy into a pinned, pooled output Frame.
 */

#include <libfreenect2/rgb_packet_processor.h>
#include <libfreenect2/allocator.h>
#include <libfreenect2/logging.h>

#include <cuda_runtime.h>
#include <nvjpeg.h>

#define CHECK_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); return false; } } while(0)
#define CALL_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); } } while(0)
#define CHECK_NVJPEG(expr) do { nvjpegStatus_t st = (expr); if (st != NVJPEG_STATUS_SUCCESS) { LOG_ERROR << #expr ": nvjpeg status " << (int)st; return false; } } while(0)
#define CALL_NVJPEG(expr) do { nvjpegStatus_t st = (expr); if (st != NVJPEG_STATUS_SUCCESS) { LOG_ERROR << #expr ": nvjpeg status " << (int)st; } } while(0)

namespace libfreenect2
{

static const int NVJPEG_WIDTH = 1920;
static const int NVJPEG_HEIGHT = 1080;

/* Expand interleaved BGR (as decoded by nvJPEG) to BGRX to match Frame::BGRX. */
__global__ void bgrToBgrxKernel(const unsigned char *__restrict__ bgr, uchar4 *__restrict__ bgrx, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const unsigned char *p = bgr + 3 * i;
  bgrx[i] = make_uchar4(p[0], p[1], p[2], 0xff);
}

/* Frame whose memory comes from a (pooled) pinned allocator. */
class NvJpegFrame: public Frame
{
public:
  NvJpegFrame(Buffer *buffer):
    Frame(NVJPEG_WIDTH, NVJPEG_HEIGHT, 4, (unsigned char*)-1)
  {
    data = buffer->data;
    rawdata = reinterpret_cast<unsigned char *>(buffer);
  }

  virtual ~NvJpegFrame()
  {
    Buffer *buffer = reinterpret_cast<Buffer*>(rawdata);
    buffer->allocator->free(buffer);
    rawdata = NULL;
  }
};

/* Pinned host allocator. Unlike the depth input buffers this is not
 * write-combined: nvJPEG parses the bitstream (and for the hybrid backends
 * does Huffman decoding) on the CPU, and CPU reads from WC memory are slow. */
class NvJpegPinnedAllocator: public Allocator
{
  bool allocate_pinned(Buffer *b, size_t size)
  {
    CHECK_CUDA(cudaHostAlloc(&b->data, size, cudaHostAllocPortable));
    b->length = 0;
    b->capacity = size;
    return true;
  }

public:
  virtual Buffer *allocate(size_t size)
  {
    Buffer *b = new Buffer();
    if (!allocate_pinned(b, size))
      b->data = NULL;
    return b;
  }

  virtual void free(Buffer *b)
  {
    if (b == NULL)
      return;
    if (b->data)
      CALL_CUDA(cudaFreeHost(b->data));
    delete b;
  }
};

class NvJpegRgbPacketProcessorImpl: public WithPerfLogging
{
public:
  static const size_t PIXELS = (size_t)NVJPEG_WIDTH * NVJPEG_HEIGHT;

  bool good;

  cudaStream_t stream;

  nvjpegHandle_t handle;
  nvjpegJpegDecoder_t decoder;
  nvjpegJpegState_t state;
  nvjpegBufferPinned_t pinned_buffer;
  nvjpegBufferDevice_t device_buffer;
  nvjpegJpegStream_t jpeg_stream;
  nvjpegDecodeParams_t decode_params;
  nvjpegBackend_t backend;

  unsigned char *d_bgr;   // 1920x1080x3, decoder output
  uchar4 *d_bgrx;         // 1920x1080x4, expanded

  Allocator *input_allocator;
  Allocator *frame_allocator;
  Frame *frame;

  NvJpegRgbPacketProcessorImpl():
    good(false),
    stream(0),
    handle(0),
    decoder(0),
    state(0),
    pinned_buffer(0),
    device_buffer(0),
    jpeg_stream(0),
    decode_params(0),
    backend(NVJPEG_BACKEND_DEFAULT),
    d_bgr(NULL),
    d_bgrx(NULL),
    input_allocator(NULL),
    frame_allocator(NULL),
    frame(NULL)
  {
    good = initialize();
    if (!good)
      return;

    input_allocator = new PoolAllocator(new NvJpegPinnedAllocator());
    frame_allocator = new PoolAllocator(new NvJpegPinnedAllocator());
    newFrame();
  }

  ~NvJpegRgbPacketProcessorImpl()
  {
    delete frame;
    delete input_allocator;
    delete frame_allocator;

    if (decode_params) CALL_NVJPEG(nvjpegDecodeParamsDestroy(decode_params));
    if (jpeg_stream) CALL_NVJPEG(nvjpegJpegStreamDestroy(jpeg_stream));
    if (state) CALL_NVJPEG(nvjpegJpegStateDestroy(state));
    if (pinned_buffer) CALL_NVJPEG(nvjpegBufferPinnedDestroy(pinned_buffer));
    if (device_buffer) CALL_NVJPEG(nvjpegBufferDeviceDestroy(device_buffer));
    if (decoder) CALL_NVJPEG(nvjpegDecoderDestroy(decoder));
    if (handle) CALL_NVJPEG(nvjpegDestroy(handle));
    if (d_bgr) CALL_CUDA(cudaFree(d_bgr));
    if (d_bgrx) CALL_CUDA(cudaFree(d_bgrx));
    if (stream) CALL_CUDA(cudaStreamDestroy(stream));
  }

  bool initialize()
  {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    {
      LOG_ERROR << "nvJPEG: no CUDA device";
      return false;
    }

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, NVJPEG_FLAGS_DEFAULT, &handle));

    /* GPU_HYBRID does Huffman decoding on the GPU; plain HYBRID would keep it on the CPU. */
    backend = NVJPEG_BACKEND_GPU_HYBRID;
    if (nvjpegDecoderCreate(handle, backend, &decoder) != NVJPEG_STATUS_SUCCESS)
    {
      LOG_WARNING << "nvJPEG: GPU_HYBRID backend unavailable, falling back to HYBRID";
      backend = NVJPEG_BACKEND_HYBRID;
      CHECK_NVJPEG(nvjpegDecoderCreate(handle, backend, &decoder));
    }

    CHECK_NVJPEG(nvjpegDecoderStateCreate(handle, decoder, &state));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle, NULL, &pinned_buffer));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle, NULL, &device_buffer));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(state, pinned_buffer));
    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(state, device_buffer));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &jpeg_stream));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &decode_params));
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decode_params, NVJPEG_OUTPUT_BGRI));

    CHECK_CUDA(cudaMalloc(&d_bgr, PIXELS * 3));
    CHECK_CUDA(cudaMalloc(&d_bgrx, PIXELS * 4));

    LOG_INFO << "nvJPEG decoder initialized (backend " << (backend == NVJPEG_BACKEND_GPU_HYBRID ? "GPU_HYBRID" : "HYBRID") << ")";
    return true;
  }

  void newFrame()
  {
    frame = new NvJpegFrame(frame_allocator->allocate(PIXELS * 4));
    frame->format = Frame::BGRX;
  }

  bool decode(const unsigned char *jpeg, size_t length)
  {
    CHECK_NVJPEG(nvjpegJpegStreamParse(handle, jpeg, length, 0, 0, jpeg_stream));

    unsigned int w = 0, h = 0;
    CHECK_NVJPEG(nvjpegJpegStreamGetFrameDimensions(jpeg_stream, &w, &h));
    if (w != (unsigned)NVJPEG_WIDTH || h != (unsigned)NVJPEG_HEIGHT)
    {
      LOG_ERROR << "nvJPEG: unexpected image size " << w << "x" << h;
      return false;
    }

    nvjpegImage_t out;
    for (int c = 0; c < NVJPEG_MAX_COMPONENT; ++c)
    {
      out.channel[c] = NULL;
      out.pitch[c] = 0;
    }
    out.channel[0] = d_bgr;
    out.pitch[0] = NVJPEG_WIDTH * 3;

    CHECK_NVJPEG(nvjpegDecodeJpegHost(handle, decoder, state, decode_params, jpeg_stream));
    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle, decoder, state, jpeg_stream, stream));
    CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle, decoder, state, &out, stream));

    const int block = 256;
    const int grid = (int)((PIXELS + block - 1) / block);
    bgrToBgrxKernel<<<grid, block, 0, stream>>>(d_bgr, d_bgrx, (int)PIXELS);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(frame->data, d_bgrx, PIXELS * 4, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return true;
  }
};

NvJpegRgbPacketProcessor::NvJpegRgbPacketProcessor():
  impl_(new NvJpegRgbPacketProcessorImpl())
{
}

NvJpegRgbPacketProcessor::~NvJpegRgbPacketProcessor()
{
  delete impl_;
}

bool NvJpegRgbPacketProcessor::good()
{
  return impl_->good;
}

void NvJpegRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ == NULL || !impl_->good)
    return;

  impl_->startTiming();

  impl_->frame->timestamp = packet.timestamp;
  impl_->frame->sequence = packet.sequence;
  impl_->frame->exposure = packet.exposure;
  impl_->frame->gain = packet.gain;
  impl_->frame->gamma = packet.gamma;

  bool ok = impl_->decode(packet.jpeg_buffer, packet.jpeg_buffer_length);

  impl_->stopTiming(LOG_INFO);

  if (!ok)
  {
    LOG_ERROR << "Failed to decompress rgb image with nvJPEG";
    return;
  }

  if (listener_->onNewFrame(Frame::Color, impl_->frame))
    impl_->newFrame();
}

Allocator *NvJpegRgbPacketProcessor::getAllocator()
{
  return impl_->good ? impl_->input_allocator : RgbPacketProcessor::getAllocator();
}

} /* namespace libfreenect2 */
