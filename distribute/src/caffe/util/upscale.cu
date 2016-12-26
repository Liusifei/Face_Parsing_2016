/*
 * upscale.cu
 *
 *      Author: Alan_Huang
 */


#include "caffe/util/upscale.hpp"
#include "device_functions.h"

namespace caffe {

/*
 *  nthreads should be n*c*4
 */
template <typename IterpolateAction, typename Dtype>
__global__ void kernel_upscale_2x_corner(const int nthreads, Dtype* src,
    const int src_h, const int src_w, Dtype* dst) {
  const int src_spatial_dim = src_h * src_w;
  const int dst_spatial_dim = src_spatial_dim * 4;
  const int dst_h = src_h * 2;
  const int dst_w = src_w * 2;
  int dst_offset[] = {0, dst_w - 1, dst_w * (dst_h -1), dst_w * dst_h - 1};
  int src_offset[] = {0, src_w - 1, src_w * (src_h -1), src_w * src_h - 1};
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c_id = index / 4;
    IterpolateAction::DoEltwise(src,
        c_id * src_spatial_dim + src_offset[index % 4],
        dst,
        c_id * dst_spatial_dim + dst_offset[index % 4]);
  }
}

/*
 *  upscale_all_border_horizontal_lines.
 *  nthreads should be n*c*(dst_w -2) * 2
 */
template <typename IterpolateAction, typename Dtype>
__global__ void kernel_upscale_2x_border_line_horizontal(const int nthreads,
    Dtype* src, const int src_h, const int src_w, Dtype* dst) {
  const int src_spatial_dim = src_h * src_w;
  const int dst_spatial_dim = src_spatial_dim * 4;
  const int dst_h = src_h * 2;
  const int dst_w = src_w * 2;
  int dst_offset[] = {0, dst_w * (dst_h -1)};
  int src_offset[] = {0, src_w * (src_h -1)};
  __shared__ Dtype zero;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c_id = index / ((dst_w -2) * 2);
    int line_id = (index / (dst_w -2)) % 2;
    int dst_w_id = 1 + (index % (dst_w -2));
    Dtype* src_p11 = src + c_id * src_spatial_dim +
        src_offset[line_id] + (dst_w_id-1)/2;
    Dtype* dst_p = dst + c_id * dst_spatial_dim +
        dst_offset[line_id] + dst_w_id;
    IterpolateAction::template Do<Dtype, 1>(src_p11,
            src_p11 + 1, &zero, &zero, dst_p,
            256/4 + 128 * ((dst_w_id-1)%2), 0);
  }
}


/*
 *  upscale_all_border_horizontal_lines.
 *  nthreads should be n*c*(dst_h -2) * 2
 */
template <typename IterpolateAction, typename Dtype>
__global__ void kernel_upscale_2x_border_line_vertical(const int nthreads,
    Dtype* src, const int src_h, const int src_w, Dtype* dst) {
  const int src_spatial_dim = src_h * src_w;
  const int dst_spatial_dim = src_spatial_dim * 4;
  const int dst_h = src_h * 2;
  const int dst_w = src_w * 2;
  int dst_offset[] = {0, dst_w - 1};
  int src_offset[] = {0, src_w - 1};
  __shared__ Dtype zero ;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c_id = index / ((dst_h -2) * 2);
    int id_inside_c = index % ((dst_h -2) * 2);
    int dst_h_id = id_inside_c / 2 + 1;
    int col_id = id_inside_c % 2 ;
    Dtype* src_p11 = src + c_id * src_spatial_dim +
        src_offset[col_id] + (dst_h_id-1)/2 * src_w;
    Dtype* dst_p = dst + c_id * dst_spatial_dim +
        dst_offset[col_id] + dst_h_id * dst_w;
    IterpolateAction::template Do<Dtype, 1>(src_p11,
          &zero, src_p11 + src_w, &zero, dst_p,
          0, 256/4 + 128 * ((dst_h_id-1)%2));
  }
}

/*
 *  upscale_all_border_horizontal_lines.
 *  nthreads should be n*c*(dst_h -2) * (dst_w -2)
 */
template <typename IterpolateAction, typename Dtype>
__global__ void kernel_upscale_2x_lines(const int nthreads,
    Dtype* src, const int src_h, const int src_w, Dtype* dst) {
  const int src_spatial_dim = src_h * src_w;
  const int dst_spatial_dim = src_spatial_dim * 4;
  const int dst_h = src_h * 2;
  const int dst_w = src_w * 2;

  CUDA_KERNEL_LOOP(index, nthreads) {
    int c_id = index / ((dst_h -2) * (dst_w -2));
    int id_inside_c = index % ((dst_h -2) * (dst_w -2));
    int dst_h_id = 1 + id_inside_c / (dst_w -2);
    int dst_w_id = 1 + id_inside_c % (dst_w -2);
    Dtype* src_p11 = src + c_id * src_spatial_dim +
        (dst_h_id-1)/2 * src_w + (dst_w_id-1)/2;
    Dtype* dst_p = dst + c_id * dst_spatial_dim +
        dst_h_id * dst_w + dst_w_id;
    IterpolateAction::template Do<Dtype, 1>(src_p11,
        src_p11 + 1, src_p11 + src_w, src_p11 + src_w + 1,
        dst_p, 256/4 + 128 * ((dst_w_id-1)%2),
        256/4 + 128 * ((dst_h_id-1)%2));
  }
}



template <typename IterpolateAction, typename Dtype>
void upscale_2x_gpu(Dtype* src_data, const int src_n, const int src_c,
    const int src_h, const int src_w, Dtype* dst_data) {
  int total_channel_num = src_n * src_c;
  int dst_h = src_h * 2;
  int dst_w = src_w * 2;
  kernel_upscale_2x_corner<IterpolateAction, Dtype> <<<
      CAFFE_GET_BLOCKS(total_channel_num * 4),
      CAFFE_CUDA_NUM_THREADS  >>> (total_channel_num * 4, src_data,
          src_h, src_w, dst_data);
  if (dst_w -2 > 0) {
    kernel_upscale_2x_border_line_horizontal<IterpolateAction, Dtype> <<<
        CAFFE_GET_BLOCKS(total_channel_num * (dst_w -2) * 2),
        CAFFE_CUDA_NUM_THREADS  >>> (total_channel_num * (dst_w -2) * 2,
            src_data, src_h, src_w, dst_data);
  }
  if (dst_h -2 > 0) {
    kernel_upscale_2x_border_line_vertical<IterpolateAction, Dtype> <<<
        CAFFE_GET_BLOCKS(total_channel_num * (dst_h -2) * 2),
        CAFFE_CUDA_NUM_THREADS  >>> (total_channel_num * (dst_h -2) * 2,
            src_data, src_h, src_w, dst_data);
  }
  if (dst_w -2 > 0 && dst_h -2 > 0) {
    kernel_upscale_2x_lines<IterpolateAction, Dtype> <<<
        CAFFE_GET_BLOCKS(total_channel_num * (dst_h -2) * (dst_w -2)),
        CAFFE_CUDA_NUM_THREADS  >>> (total_channel_num * (dst_h -2) * (dst_w -2),
            src_data, src_h, src_w, dst_data);
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void Blob2xUpscaler<Dtype>::Forward_gpu(const Blob<Dtype>& src_blob,
    Blob<Dtype>& dst_blob) {
  Blob2xUpscaler<Dtype>::Check(src_blob, dst_blob);
  int last_dim = src_blob.shape().size() - 1;
  int total_channel_num = src_blob.count(0, src_blob.shape().size() - 2);
  int src_spatial_dim = src_blob.count(last_dim - 1);
  int dst_spatial_dim = dst_blob.count(last_dim - 1);
  int src_h = src_blob.shape(last_dim - 1);
  int src_w = src_blob.shape(last_dim);

  Dtype* src_data = const_cast<Dtype*>(src_blob.gpu_data());
  Dtype* dst_data = dst_blob.mutable_gpu_data();

  upscale_2x_gpu<PointInterpolateForward, Dtype>(src_data, total_channel_num,
      1, src_h, src_w, dst_data);
}


template <typename Dtype>
void Blob2xUpscaler<Dtype>::Backward_gpu(const Blob<Dtype>& dst_blob,
    Blob<Dtype>& src_blob) {

  Blob2xUpscaler<Dtype>::Check(src_blob, dst_blob);
  int last_dim = src_blob.shape().size() - 1;
  int total_channel_num = src_blob.count(0, src_blob.shape().size() - 2);
  int src_spatial_dim = src_blob.count(last_dim - 1);
  int dst_spatial_dim = dst_blob.count(last_dim - 1);
  int src_h = src_blob.shape(last_dim - 1);
  int src_w = src_blob.shape(last_dim);

  Dtype* dst_data = const_cast<Dtype*>(dst_blob.gpu_diff());
  Dtype* src_data = src_blob.mutable_gpu_diff();

  upscale_2x_gpu<PointInterpolateBackward, Dtype>(src_data, total_channel_num,
        1, src_h, src_w, dst_data);
}



template void Blob2xUpscaler<float>::Forward_gpu(const Blob<float>& src_blob,
    Blob<float>& dst_blob);
template void Blob2xUpscaler<double>::Forward_gpu(const Blob<double>& src_blob,
    Blob<double>& dst_blob);

template void Blob2xUpscaler<float>::Backward_gpu(const Blob<float>& dst_blob,
    Blob<float>& src_blob);
template void Blob2xUpscaler<double>::Backward_gpu(const Blob<double>& dst_blob,
    Blob<double>& src_blob);


}  // namespace caffe

