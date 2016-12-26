#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/elementhingeloss_layer.hpp"

namespace caffe {

template <typename Dtype>
void ElementHingeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  /*const bool need_scale = (this->layer_param_.has_element_hinge_loss_param() &&
  		  this->layer_param_.element_hinge_loss_param().normalize_per_positive());*/
  const bool need_scale = this->layer_param_.element_hinge_loss_param().scale_loss();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int num = bottom[0]->num();

  Dtype* loss = top[0]->mutable_cpu_data();

  caffe::caffe_copy(count,bottom[1]->gpu_data(),sign_.mutable_gpu_data());
  caffe::caffe_gpu_add_scalar(count,Dtype(-0.5),sign_.mutable_gpu_data());
  caffe::caffe_gpu_sign(count,sign_.mutable_gpu_data(),sign_.mutable_gpu_data());
  caffe::caffe_gpu_scal(count,Dtype(-1),sign_.mutable_gpu_data());

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      bottom_diff);

  caffe::caffe_gpu_mul(count,sign_.gpu_data(),bottom_diff,bottom_diff);
  caffe::caffe_gpu_scalar_max(count,Dtype(0),bottom_diff,bottom_diff);

  this->scale_factor_ = 1;
  switch (this->layer_param_.element_hinge_loss_param().norm()) {
    case ElementHingeLossParameter_Norm_L1:
      caffe_gpu_asum(count, bottom_diff,loss);
      loss[0]/= num;
      break;
    case ElementHingeLossParameter_Norm_L2:
      caffe_gpu_dot(count, bottom_diff, bottom_diff,loss);
      loss[0]/=  num*Dtype(2);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }

  if(need_scale)
    {
	  /*const int gt_bottom_id =  this->layer_param_.element_hinge_loss_param().label_bottom_id();
	  CHECK(gt_bottom_id < bottom.size());
	  caffe::caffe_gpu_set(count,Dtype(1),one_.mutable_gpu_data());
	  caffe::caffe_gpu_sign(count,bottom[gt_bottom_id]->gpu_data(),one_.mutable_gpu_diff());
	  caffe::caffe_gpu_abs(count,one_.gpu_diff(),one_.mutable_gpu_diff());
	  Dtype sum;
	  caffe_gpu_dot(count,
			  one_.gpu_diff(), one_.gpu_data(),&sum);
	  this->scale_factor_ = ((sum+1))/ bottom[0]->num();*/
      this->scale_factor_ = bottom[0]->height()*bottom[0]->width();
	  loss[0] /= scale_factor_;

    }


}

template <typename Dtype>
void ElementHingeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
			   << " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		int num = bottom[0]->num();
		int count = bottom[0]->count();
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe::caffe_gpu_mul(count,sign_.gpu_data(),bottom_diff,bottom_diff);

		switch (this->layer_param_.element_hinge_loss_param().norm()) {
			case ElementHingeLossParameter_Norm_L1:
			  caffe_gpu_sign(count, bottom_diff, bottom_diff);
			  caffe_gpu_scal(count, loss_weight / num/scale_factor_, bottom_diff);
			  break;
			case ElementHingeLossParameter_Norm_L2:
			  caffe_gpu_scal(count, loss_weight * 2 / num/scale_factor_, bottom_diff);
			  break;
			default:
			  LOG(FATAL) << "Unknown Norm";
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(ElementHingeLossLayer);
}  // namespace caffe
