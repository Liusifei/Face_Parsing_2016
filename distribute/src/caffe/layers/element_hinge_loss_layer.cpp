#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/elementhingeloss_layer.hpp"

namespace caffe {

template <typename Dtype>
void ElementHingeLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  sign_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  one_.Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ElementHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  /*const bool need_scale = (this->layer_param_.has_element_hinge_loss_param() &&
		  this->layer_param_.element_hinge_loss_param().normalize_per_positive());*/
  const bool need_scale = this->layer_param_.element_hinge_loss_param().scale_loss();

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int num = bottom[0]->num();



  caffe::caffe_copy(count,bottom[1]->cpu_data(),sign_.mutable_cpu_data());
  caffe::caffe_add_scalar(count,Dtype(-0.5),sign_.mutable_cpu_data());
  caffe::caffe_cpu_sign(count,sign_.mutable_cpu_data(),sign_.mutable_cpu_data());
  caffe::caffe_scal(count,Dtype(-1),sign_.mutable_cpu_data());

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      bottom_diff);

  caffe::caffe_mul(count,sign_.cpu_data(),bottom_diff,bottom_diff);
  caffe::caffe_scalar_max(count,Dtype(0),bottom_diff,bottom_diff);

  Dtype loss = 0;
  switch (this->layer_param_.element_hinge_loss_param().norm()) {
    case ElementHingeLossParameter_Norm_L1:
      loss = caffe_cpu_asum(count, bottom_diff) / num;
      break;
    case ElementHingeLossParameter_Norm_L2:
      loss = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num  / Dtype(2);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  this->scale_factor_ = 1;
  if(need_scale){
	  /*const int gt_bottom_id =  this->layer_param_.element_hinge_loss_param().label_bottom_id();
	  CHECK(gt_bottom_id < bottom.size());
	  caffe::caffe_set(count,Dtype(1),one_.mutable_cpu_data());
  	  caffe::caffe_cpu_sign(count,bottom[gt_bottom_id]->cpu_data(),one_.mutable_cpu_diff());
  	  caffe::caffe_abs(count,one_.cpu_diff(),one_.mutable_cpu_diff());
  	  Dtype sum = caffe_cpu_dot(count,
  			  one_.cpu_diff(), one_.cpu_data()) +1;
              */
  	  this->scale_factor_ = bottom[0]->height()*bottom[0]->width();
  	  loss /= scale_factor_;
  }
  top[0]->mutable_cpu_data()[0] = loss;


}

template <typename Dtype>
void ElementHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
			   << " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		int num = bottom[0]->num();
		int count = bottom[0]->count();
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe::caffe_mul(count,sign_.cpu_data(),bottom_diff,bottom_diff);

		switch (this->layer_param_.element_hinge_loss_param().norm()) {
			case ElementHingeLossParameter_Norm_L1:
			  caffe_cpu_sign(count, bottom_diff, bottom_diff);
			  caffe_scal(count, loss_weight / num/scale_factor_, bottom_diff);
			  break;
			case ElementHingeLossParameter_Norm_L2:
			  caffe_scal(count, loss_weight * 2 / num/scale_factor_, bottom_diff);
			  break;
			default:
			  LOG(FATAL) << "Unknown Norm";
		}
	}

}

#ifdef CPU_ONLY
STUB_GPU(ElementHingeLossLayer);
#endif

INSTANTIATE_CLASS(ElementHingeLossLayer);
REGISTER_LAYER_CLASS(ElementHingeLoss);


}  // namespace caffe
