
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/util_img.hpp"

namespace caffe {

template <typename Dtype>
void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ResizeParameter resize_param = this->layer_param_.resize_param();
 // CHECK(resize_param.has_resize_ratio()) << "Resize ratio (A positive number) is required.";


  
  resize_ratio_=1.0;

  if(resize_param.has_resize_ratio())
    resize_ratio_ = static_cast<Dtype>(resize_param.resize_ratio());
  CHECK_GT(resize_ratio_, 0);
  
  for(int i=0;i<4; i++)
  {
	  this->locs_.push_back(new Blob<Dtype>);
  }
  

}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //resized_height_ = static_cast<int>(height_ * resize_ratio_);
  //resized_width_ = static_cast<int>(width_ * resize_ratio_);
  ResizeParameter resize_param = this->layer_param_.resize_param();
  out_num_  = bottom[0]->num();
  out_channels_ = channels_;



  bool is_pyramid_test = resize_param.is_pyramid_test();
  if(is_pyramid_test == false)
  {
	if(resize_param.has_height())
	{

	  this->out_height_ = resize_param.height();
	  this->out_width_ = resize_param.width();
	}
	else
	{
		this->out_height_ = static_cast<int>(height_ * resize_ratio_);
  		this->out_width_ = static_cast<int>(width_ * resize_ratio_);
	}
  }
  else
  {
	  int in_height = bottom[0]->height();
	  int in_width = bottom[0]->width();
	  this->out_height_ = static_cast<int> (resize_param.out_height_scale() * in_height);
	  this->out_width_ = static_cast<int> (resize_param.out_width_scale() * in_width);
  }


  resized_height_ = this->out_height_;
  resized_width_ = this->out_width_;
  
  top[0]->Reshape(bottom[0]->num(), channels_, resized_height_,
      resized_width_);
  for(int i=0;i<4; ++i)
  {
	  this->locs_[i]->Reshape(1,1,resized_height_, resized_width_);
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
    switch (this->layer_param_.resize_param().type()) 
    {
        case ResizeParameter_Type_BILINEAR:
            ResizeBlob_cpu(bottom[0],top[0] );
            break;
        case ResizeParameter_Type_NEAREST:
          for (int n = 0; n < bottom[0]->num(); ++n) {
            for (int c = 0; c < channels_; ++c) {
              for (int rh = 0; rh < resized_height_; ++rh) {
                for (int rw = 0; rw < resized_width_; ++rw) {
                  int h = int(rh / resize_ratio_);
                  int w = int(rw / resize_ratio_);
                  h = std::min(h, height_);
                  w = std::min(w, width_);
                  top_data[rh * resized_width_ + rw] = 
                    bottom_data[h * width_ + w];
                }
              }
              // compute offset
              bottom_data += bottom[0]->offset(0, 1);
              top_data += top[0]->offset(0, 1);
            }
          }
          break;
        default:
            LOG(FATAL) << "Unknown resize type.";
    }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  
    const Dtype* loc1 = this->locs_[0]->cpu_data();
	const Dtype* weight1 = this->locs_[0]->cpu_diff();
	const Dtype* loc2 = this->locs_[1]->cpu_data();
	const Dtype* weight2 = this->locs_[1]->cpu_diff();
	const Dtype* loc3 = this->locs_[2]->cpu_data();
	const Dtype* weight3 = this->locs_[2]->cpu_diff();
	const Dtype* loc4 = this->locs_[3]->cpu_data();
	const Dtype* weight4 = this->locs_[3]->cpu_diff();
    
    
    switch (this->layer_param_.resize_param().type()) 
    {
        case ResizeParameter_Type_BILINEAR:
            caffe::GetBiLinearResizeMatRules_cpu( bottom[0]->height(),bottom[0]->width(),
			top[0]->height(), top[0]->width(),
			this->locs_[0]->mutable_cpu_data(), this->locs_[0]->mutable_cpu_diff(),
			this->locs_[1]->mutable_cpu_data(), this->locs_[1]->mutable_cpu_diff(),
			this->locs_[2]->mutable_cpu_data(), this->locs_[2]->mutable_cpu_diff(),
			this->locs_[3]->mutable_cpu_data(), this->locs_[3]->mutable_cpu_diff() );
            for(int n=0; n< this->out_num_; ++n)
            {
                for(int c = 0; c < this->out_channels_; ++c)
                {
                    int bottom_diff_offset = bottom[0]->offset(n,c);
                    int top_diff_offset = top[0]->offset(n,c);

                    for (int idx = 0; idx < this->out_height_* this->out_width_; ++idx)
                    {
                        bottom_diff[bottom_diff_offset + static_cast<int>(loc1[idx])] += top_diff[top_diff_offset+idx]*weight1[idx];
                        bottom_diff[bottom_diff_offset + static_cast<int>(loc2[idx])] += top_diff[top_diff_offset+idx]*weight2[idx];
                        bottom_diff[bottom_diff_offset + static_cast<int>(loc3[idx])] += top_diff[top_diff_offset+idx]*weight3[idx];
                        bottom_diff[bottom_diff_offset + static_cast<int>(loc4[idx])] += top_diff[top_diff_offset+idx]*weight4[idx];
                    }
                }
            }
            break;
        case ResizeParameter_Type_NEAREST:
            for (int n = 0; n < top[0]->num(); ++n) {
                for (int c = 0; c < channels_; ++c) {
                  for (int h = 0; h < height_; ++h) {
                    for (int w = 0; w < width_; ++w) {
                      int hstart = int(h * resize_ratio_);
                      int wstart = int(w * resize_ratio_);
                      int hend = int(hstart + resize_ratio_);
                      int wend = int(wstart + resize_ratio_);
                      hstart = std::max(hstart, 0);
                      wstart = std::max(wstart, 0);
                      hend = std::min(hend, resized_height_);
                      wend = std::min(wend, resized_width_);
                      for (int rh = hstart; rh < hend; ++rh) {
                        for (int rw = wstart; rw < wend; ++rw) {
                          bottom_diff[h * width_ + w] +=
                            top_diff[rh * resized_width_ + rw];
                        }
                      }
                    }
                  }
                  // offset
                  bottom_diff += bottom[0]->offset(0, 1);
                  top_diff += top[0]->offset(0, 1);
                }
            }
            break;
        
        default:
            LOG(FATAL) << "Unknown resize type.";
    }
        
  
}

#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe
