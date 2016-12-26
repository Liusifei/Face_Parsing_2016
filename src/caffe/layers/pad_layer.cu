/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layers/pad_layer.hpp"

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void copy_kernel2(const int n, const int copycount,
    const int src_line_count, const int dest_line_count,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index * src_line_count;      
    int dest_start = index * dest_line_count;
    for (int i = 0; i < copycount; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}



// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void copy_kernel(const int n, const int height, const int width,
    const int src_outer_stride, const int src_inner_stride,
    const int dest_outer_stride, const int dest_inner_stride,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, n) {
    int src_start = index / height * src_outer_stride
                  + index % height * src_inner_stride;
    int dest_start = index / height * dest_outer_stride
                   + index % height * dest_inner_stride;
    for (int i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int lines = bottom[0]->count() / bottom[0]->width();
  const int copycount = bottom[0]->width();

  caffe_gpu_set(top[0]->count(), static_cast<Dtype>(0), top_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  copy_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
      lines, bottom[0]->height(), bottom[0]->width(),
      bottom[0]->height() * bottom[0]->width(), bottom[0]->width(),
      top[0]->height() * top[0]->width(), top[0]->width(),
      bottom_data , top_data + top[0]->offset(0, 0, start_h_, start_w_));

  /*copy_kernel2<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(lines,copycount,bottom[0]->width(),top[0]->width(),bottom_data,top_data + top[0]->offset(0, 0, start_h_, start_w_));*/
}

template <typename Dtype>
void PadLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int lines = bottom[0]->count() / bottom[0]->width();
  const int copycount = bottom[0]->width();

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    copy_kernel<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(
        lines, bottom[0]->height(), bottom[0]->width(),
        top[0]->height() * top[0]->width(), top[0]->width(),
        bottom[0]->height() * bottom[0]->width(), bottom[0]->width(),
        top_diff+ top[0]->offset(0, 0, start_h_, start_w_), bottom_diff);
    /*copy_kernel2<<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(lines,copycount,top[0]->width(),bottom[0]->width(),top_diff + top[0]->offset(0, 0, start_h_, start_w_),bottom_diff);*/
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PadLayer);

}  // namespace caffe
