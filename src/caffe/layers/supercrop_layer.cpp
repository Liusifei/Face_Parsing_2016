/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/supercrop_layer.hpp"

namespace caffe {

template <typename Dtype>
void SuperCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  /*this layer is DIFFERENT from FCN caffe code. 
    I just implement simple cropping from center point. 
    Do let me know if you find FCN's strictly crop is critical. liangji,20150407.
  */
  //CHECK(bottom[1]->height() <= bottom[0]->height())<<"bottom 1 size should smaller than bottom 0";
  //CHECK(bottom[1]->width() <= bottom[0]->width())<<"bottom 1 size should smaller than bottom 0";
  
  //crop_h_ = int((bottom[0]->height() - bottom[1]->height())/2);
  //crop_w_ = int((bottom[0]->width() - bottom[1]->width())/2);
  crop_h_ = this->layer_param_.super_crop_param().crop_h();
  crop_w_ = this->layer_param_.super_crop_param().crop_w();
  if(bottom.size()>1 && this->layer_param_.super_crop_param().type() != SuperCropParameter_Type_ONEPOINT && this->layer_param_.super_crop_param().type() != SuperCropParameter_Type_TWOPOINT)
  {
    CHECK(bottom.size()==2);
    crop_h_ = bottom[1]->height();
    crop_w_ = bottom[1]->width();
  }
  CHECK(crop_h_ > 0 && crop_w_ >0);
  CHECK(crop_h_ <= bottom[0]->height())<<"crop h should smaller than bottom height";
  CHECK(crop_w_ <= bottom[0]->width())<<"crop w should smaller than bottom width";
}

template <typename Dtype>
void SuperCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

   top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), crop_h_,
      crop_w_);
      
      
   const Dtype* point_data = bottom[1]->cpu_data();
   int cx,cy,x1,y1,x2,y2;

    switch (this->layer_param_.super_crop_param().type()) 
    {
        case SuperCropParameter_Type_CENTER:
            start_w_ = int((bottom[0]->width() - crop_w_)/2);
            start_h_ = int((bottom[0]->height() - crop_h_)/2);
            break;
        case SuperCropParameter_Type_RANDOM:
            start_w_ = int(((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1)) * (float(bottom[0]->width() - crop_w_)));
            start_h_ = int(((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1)) * (float(bottom[0]->height() - crop_h_)));
            break;
	case SuperCropParameter_Type_ONEPOINT:
	    CHECK(bottom[1]->channels()==2);
	    cx = point_data[0];
	    cy = point_data[1];
            start_w_ = cx - crop_w_/2 + this->layer_param_.super_crop_param().point_fix_w();
            start_h_ = cy - crop_h_/2 + this->layer_param_.super_crop_param().point_fix_h();
	    //start_w_ = int(((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1)) * (float(bottom[0]->width() - crop_w_)));
            //start_h_ = int(((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1)) * (float(bottom[0]->height() - crop_h_)));
	    break;
	case SuperCropParameter_Type_TWOPOINT:
	    CHECK(bottom[1]->channels()==4);
		x1 = point_data[0];
		y1 = point_data[1];
		x2 = point_data[2];
		y2 = point_data[3];
	    cx = (x1+x2)/2;
	    cy = (y1+y2)/2;
            start_w_ = cx - crop_w_/2 + this->layer_param_.super_crop_param().point_fix_w();
            start_h_ = cy - crop_h_/2 + this->layer_param_.super_crop_param().point_fix_h();

	    break;
        default:
                LOG(FATAL) << "Unknown type method.";
    }
    start_w_ = std::max(0,start_w_);
    start_w_ = std::min(bottom[0]->width() - crop_w_,start_w_);
    start_h_ = std::max(0,start_h_);
    start_h_ = std::min(bottom[0]->height() - crop_h_,start_h_);
    CHECK(start_w_>=0);
    CHECK(start_h_>=0);
    CHECK(start_w_ + crop_w_ <= bottom[0]->width());
    CHECK(start_h_ + crop_h_ <= bottom[0]->height());
    if(this->layer_param_.super_crop_param().print_info())
    {
        LOG(INFO)<<"start_h = "<<start_h_<<", start_w = "<<start_w_<<", crop_h = "<<crop_h_<<", crop_w = "<<crop_w_;
    }
}

template <typename Dtype>
void SuperCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
    
    Dtype* top_data = top[0]->mutable_cpu_data();

    for (int n = 0; n < top[0]->num(); ++n) {
        for (int c = 0; c < top[0]->channels(); ++c) {
          for (int h = 0; h < top[0]->height(); ++h) {
            caffe_copy(top[0]->width(),
                bottom_data + bottom[0]->offset(n, c, start_h_ + h, start_w_),
                top_data + top[0]->offset(n, c, h));
          }
        }
    }
           
}

template <typename Dtype>
void SuperCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);

    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, start_h_ + h, start_w_));
        }
      }
    }
  }     
}

#ifdef CPU_ONLY
STUB_GPU(SuperCropLayer);
#endif

INSTANTIATE_CLASS(SuperCropLayer);
REGISTER_LAYER_CLASS(SuperCrop);

}  // namespace caffe
