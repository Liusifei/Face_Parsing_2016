#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/util_img.hpp"

namespace caffe {
template <typename Dtype>
__global__ void ResizeForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int resized_height, const int resized_width,
    const Dtype resize_ratio, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int rw = index % resized_width;
    int rh = (index / resized_width) % resized_height;
    int c = (index / resized_width / resized_height) % channels;
    int n = index / resized_width / resized_height / channels;
    int h = int(rh / resize_ratio);
    int w = int(rw / resize_ratio);
    bottom_data += (n * channels + c) * height * width;
    top_data[index] = bottom_data[h * width + w];
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  
    switch (this->layer_param_.resize_param().type()) 
    {
        case ResizeParameter_Type_BILINEAR:
            ResizeBlob_gpu(bottom[0],top[0] );
            break;
        case ResizeParameter_Type_NEAREST:
            // NOLINT_NEXT_LINE(whitespace/operators)
              ResizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                  count, bottom_data, bottom[0]->num(), channels_,
                  height_, width_, resized_height_, resized_width_, resize_ratio_, top_data);
              CUDA_POST_KERNEL_CHECK;
            break;
        default:
            LOG(FATAL) << "Unknown resize type.";
    }
  
}

template <typename Dtype>
__global__ void ResizeBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int resized_height, const int resized_width,
    const Dtype resize_ratio, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int hstart = int(h * resize_ratio);
    int wstart = int(w * resize_ratio);
    int hend = int(hstart + resize_ratio);
    int wend = int(wstart + resize_ratio);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, resized_height);
    wend = min(wend, resized_width);
    top_diff += (n * channels + c) * resized_height * resized_width;
    for (int rh = hstart; rh < hend; ++rh) {
      for (int rw = wstart; rw < wend; ++rw) {
        bottom_diff[index] += top_diff[rh * resized_width + rw];
      }
    }
  }
}

template <typename Dtype>
__global__ void kernel_ResizeBackward(const int nthreads, const Dtype* top_diff, const int top_step,
		Dtype* bottom_diff, const int bottom_step,
		const Dtype* loc1,const  Dtype* weight1, const Dtype* loc2, const Dtype* weight2,
		const Dtype* loc3,const Dtype* weight3, const Dtype* loc4, const Dtype* weight4)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int bottom_diff_offset = bottom_step*index;
		int top_diff_offset = top_step*index;
		for (int idx = 0; idx < top_step; ++idx)
		{
			bottom_diff[bottom_diff_offset + int(loc1[idx])] += top_diff[top_diff_offset+idx]*weight1[idx];
			bottom_diff[bottom_diff_offset + int(loc2[idx])] += top_diff[top_diff_offset+idx]*weight2[idx];
			bottom_diff[bottom_diff_offset + int(loc3[idx])] += top_diff[top_diff_offset+idx]*weight3[idx];
			bottom_diff[bottom_diff_offset + int(loc4[idx])] += top_diff[top_diff_offset+idx]*weight4[idx];
		}
	}
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  
    const Dtype* loc1 = this->locs_[0]->gpu_data();
	const Dtype* weight1 = this->locs_[0]->gpu_diff();
	const Dtype* loc2 = this->locs_[1]->gpu_data();
	const Dtype* weight2 = this->locs_[1]->gpu_diff();
	const Dtype* loc3 = this->locs_[2]->gpu_data();
	const Dtype* weight3 = this->locs_[2]->gpu_diff();
	const Dtype* loc4 = this->locs_[3]->gpu_data();
	const Dtype* weight4 = this->locs_[3]->gpu_diff();
    
    const int top_step = top[0]->offset(0,1);
    const int bottom_step = bottom[0]->offset(0,1);
    int loop_n = this->out_num_ * this->out_channels_;
  
    switch (this->layer_param_.resize_param().type()) 
    {
        case ResizeParameter_Type_BILINEAR:
            caffe::GetBiLinearResizeMatRules_gpu( bottom[0]->height(),bottom[0]->width(),
			top[0]->height(), top[0]->width(),
			this->locs_[0]->mutable_gpu_data(), this->locs_[0]->mutable_gpu_diff(),
			this->locs_[1]->mutable_gpu_data(), this->locs_[1]->mutable_gpu_diff(),
			this->locs_[2]->mutable_gpu_data(), this->locs_[2]->mutable_gpu_diff(),
			this->locs_[3]->mutable_gpu_data(), this->locs_[3]->mutable_gpu_diff() );

            

            kernel_ResizeBackward<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(
                    loop_n, top_diff, top_step,
                    bottom_diff, bottom_step,
                    loc1,weight1, loc2, weight2,
                    loc3,weight3,loc4, weight4);
            CUDA_POST_KERNEL_CHECK;
            break;
        case ResizeParameter_Type_NEAREST:
            ResizeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                  count, top_diff, top[0]->num(), channels_,
                  height_, width_, resized_height_, resized_width_, resize_ratio_, bottom_diff);
            CUDA_POST_KERNEL_CHECK;
            break;
        default:
            LOG(FATAL) << "Unknown resize type.";
    }
    
}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);


}  // namespace caffe
