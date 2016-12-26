/*
 * upscale.hpp
 *
 *      Author: Alan_Huang
 */

#ifndef CAFFE_UPSCALE_HPP_
#define CAFFE_UPSCALE_HPP_

#include "alan_core/base/expression_template.hpp"
#include "caffe/blob.hpp"

using namespace fxnet;

namespace caffe {

#ifdef __CUDACC__
static __inline__ __device__ float  AtomicAdd(float* address, float val) {
  return atomicAdd(address, val);
}
static __inline__ __device__ double AtomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +
                             __longlong_as_double(assumed)));
  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#else
template<typename Dtype>
static FXNET_CINLINE Dtype AtomicAdd(Dtype* address, Dtype val) {
  Dtype old = *address;
  *address += val;
  return old;
}

#endif

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &p){
  if(p.size() <= 0){ os << "[ ]"; return os; }
  os <<"["; for (int i = 0; i < p.size() -1 ; ++i) { os << p[i] << ", "; }
  os << p[p.size()-1]; os << "]"; return os;
}

template <typename Dtype>
std::ostream &operator<<(std::ostream &os,const Blob<Dtype> &inst);

inline bool operator == (const std::vector<int> &a,
    const std::vector<int>&b) {
  if(a.size() != b.size()) {
    return false;
  }
  for(int i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}



struct PointInterpolateForward {
  template <typename Dtype>
  FXNET_XINLINE static void DoEltwise(const Dtype* src,
      const int src_id, Dtype* dst, const int dst_id) {
    dst[dst_id] = src[src_id];
  }

  /* -> w   | h
   * [src_p11, src_p21]    [wX([0,256]) = projected_x - loc_x_src_p11]
   * [src_p12, src_p22]    [wY([0,256]) = projected_y - loc_y_src_p11]
   */
  template <typename Dtype, int channel_dim>
  FXNET_XINLINE static void Do(const Dtype* src_p11,
      const Dtype* src_p21, const Dtype* src_p12, const Dtype* src_p22,
      Dtype* dst_p, const int wX, const int wY) {
    int f24 = (wX * wY) / 256;
    int f14 = wY - f24;
    int f23 = wX - f24;
    int f13 = ((256 - wX) * (256 - wY)) / 256; // this one can be computed faster
  #ifdef __CUDACC__
      // Only CUDA compiler support this pragma. G++ does not.
      #pragma unroll
  #endif
    for (int i = 0; i < channel_dim; ++i) {
      *(dst_p + i) =  ((*(src_p11 + i)) * f13 + (*(src_p21 + i)) * f23 +
          (*(src_p12 + i)) * f14 + (*(src_p22 + i)) * f24) / 256;
    }
  }

  template <typename Dtype, int channel_dim>
  FXNET_XINLINE static void Init(const Dtype* src, const int src_h, const int src_w,
      Dtype* dst) {
    const int dst_h = src_h * 2;
    const int dst_w = src_w * 2;
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.
    #pragma unroll
#endif
    for (int i = 0; i < channel_dim; ++i) {
      dst[i] = src[i];
      dst[(dst_w - 1) * channel_dim + i ] =
          src[(src_w -1) * channel_dim + i ];
      dst[dst_w * (dst_h -1) * channel_dim + i] =
          src[src_w * (src_h -1) * channel_dim + i];
      dst[(dst_w * dst_h - 1) * channel_dim + i] =
          src[(src_w * src_h - 1) * channel_dim + i];
    }
  }
};





struct PointInterpolateBackward {
  template <typename Dtype>
  FXNET_XINLINE static void DoEltwise(Dtype* src_diff,
      const int src_id, const Dtype* dst_diff, const int dst_id) {
    AtomicAdd(src_diff + src_id , dst_diff[dst_id]);
  }
  /* -> w   | h
   * [src_p11, src_p21]    [wX([0,256]) = projected_x - loc_x_src_p11]
   * [src_p12, src_p22]    [wY([0,256]) = projected_y - loc_y_src_p11]
   */
  template <typename Dtype, int channel_dim>
  FXNET_XINLINE static void Do(Dtype* src_p11_diff,
      Dtype* src_p21_diff, Dtype* src_p12_diff, Dtype* src_p22_diff,
      const Dtype* dst_p_diff, const int wX, const int wY) {
    int f24 = (wX * wY) / 256;
    int f14 = wY - f24;
    int f23 = wX - f24;
    int f13 = ((256 - wX) * (256 - wY)) / 256; // this one can be computed faster
  #ifdef __CUDACC__
      // Only CUDA compiler support this pragma. G++ does not.
      #pragma unroll
  #endif
    for (int i = 0; i < channel_dim; ++i) {
      const Dtype grad = *(dst_p_diff + i) / 256;
      AtomicAdd(src_p11_diff + i, f13 * grad );
      AtomicAdd(src_p21_diff + i, f23 * grad );
      AtomicAdd(src_p12_diff + i, f14 * grad );
      AtomicAdd(src_p22_diff + i, f24 * grad );
    }
  }

  template <typename Dtype, int channel_dim>
  FXNET_CINLINE static void Init(Dtype* src_diff,
      const int src_h, const int src_w, const Dtype* dst_diff) {
    const int dst_h = src_h * 2;
    const int dst_w = src_w * 2;
#ifdef __CUDACC__
    // Only CUDA compiler support this pragma. G++ does not.
    #pragma unroll
#endif
    for (int i = 0; i < channel_dim; ++i) {
      src_diff[i] += dst_diff[i];
      src_diff[(src_w -1) * channel_dim + i] +=
          dst_diff[(dst_w - 1) * channel_dim + i];
      src_diff[src_w * (src_h -1) * channel_dim + i] +=
          dst_diff[dst_w * (dst_h -1) * channel_dim + i];
      src_diff[(src_w * src_h - 1) * channel_dim + i] +=
          dst_diff[(dst_w * dst_h - 1) * channel_dim + i];
    }
  }
};

/*
 * Up-scale border line around x axis. Ingore the start and end point.
 */

template <typename IterpolateAction, typename Dtype, int channel_dim>
FXNET_CINLINE void upscale_2x_border_line_horizontal(Dtype* src,
    Dtype* dst, Dtype* zero, const int dst_w) {
  for (int dst_w_id = 1; dst_w_id < dst_w -1; ++dst_w_id) {
    int src_w_id_1 =  (dst_w_id-1)/2;
    Dtype* src_p11 = src + src_w_id_1;
    IterpolateAction::template Do<Dtype, channel_dim>(src_p11,
        src_p11 + channel_dim, zero, zero, dst + dst_w_id,
        256/4 + 128 * ((dst_w_id-1)%2), 0);
  }
}

template <typename IterpolateAction,typename Dtype, int channel_dim>
FXNET_CINLINE void upscale_2x_line(Dtype* src_line1,
    Dtype* src_line2, Dtype* dst,
    const int dst_w, const int wY) {

  IterpolateAction::template Do<Dtype, channel_dim>(src_line1,
      src_line1, src_line2, src_line2,
      dst, 256, wY);

  for (int dst_w_id = 1; dst_w_id < dst_w -1; ++dst_w_id) {
    int src_w_id_1 =  (dst_w_id-1)/2;
    Dtype* src_p11 = src_line1 + src_w_id_1;
    Dtype* src_p12 = src_line2 + src_w_id_1;
    IterpolateAction::template Do<Dtype, channel_dim>(src_p11,
        src_p11 + channel_dim, src_p12, src_p12 + channel_dim,
        dst + dst_w_id, 256/4 + 128 * ((dst_w_id-1)%2), wY);
  }

  int src_w_id_1 =  (dst_w-2)/2;
  Dtype* src_p11 = src_line1 + src_w_id_1;
  Dtype* src_p12 = src_line2 + src_w_id_1;
  IterpolateAction::template Do<Dtype, channel_dim>(src_p11,
      src_p11, src_p12, src_p12, dst + dst_w -1, 256, wY);
}

template <typename IterpolateAction,typename Dtype>
inline void upscale_2x_cpu(Dtype* src, const int src_h, const int src_w,
    Dtype* dst) {
  const int dst_h = src_h * 2;
  const int dst_w = src_w * 2;
  // special case for the 4 corners.
  IterpolateAction::template Init<Dtype, 1>(src, src_h, src_w, dst);
  // special case for h = 0 or h = dst_h-1
  Dtype zero = 0;
  upscale_2x_border_line_horizontal<IterpolateAction, Dtype, 1>(
      src, dst, &zero, dst_w);
  upscale_2x_border_line_horizontal<IterpolateAction, Dtype, 1>(
      src + src_w * (src_h -1),
      dst + dst_w * (dst_h -1), &zero, dst_w);

  for (int dst_h_id = 1; dst_h_id < dst_h -1; ++dst_h_id) {
    Dtype* src_line1 = src + (dst_h_id-1)/2 * src_w;
    upscale_2x_line<IterpolateAction, Dtype, 1>(src_line1, src_line1 + src_w,
        dst + dst_h_id * dst_w, dst_w, 256/4 + 128 * ((dst_h_id-1)%2));
  }
}

template <typename Dtype>
class Blob2xUpscaler {
 public:
  static void Forward_cpu(const Blob<Dtype>& src_blob, Blob<Dtype>& dst_blob);
  static void Backward_cpu(const Blob<Dtype>& dst_blob, Blob<Dtype>& src_blob);

#ifndef CPU_ONLY
  static void Forward_gpu(const Blob<Dtype>& src_blob, Blob<Dtype>& dst_blob);
  static void Backward_gpu(const Blob<Dtype>& dst_blob, Blob<Dtype>& src_blob);
#endif

 protected:
  static void Check(const Blob<Dtype>& src_blob, const Blob<Dtype>& dst_blob);

};


}  // namespace caffe


#endif /* CAFFE_UPSCALE_HPP_ */
