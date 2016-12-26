/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_REGIONCONV_LAYER_HPP_
#define CAFFE_REGIONCONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RegionconvolutionLayer : public Layer<Dtype> {
/// RegionconvolutionLayer Layer, add by liangji, 20150424
public:
  explicit RegionconvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Regionconvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int input_patch_h_, input_patch_w_;
  int input_stride_h_, input_stride_w_;
  int input_pad_h_, input_pad_w_;
  bool input_is_1x1_;
  Blob<Dtype> input_col_buffer_;
  int input_col_offset_;
  
  int output_patch_h_, output_patch_w_;
  int output_stride_h_, output_stride_w_;
  int output_pad_h_, output_pad_w_;
  bool output_is_1x1_;
  Blob<Dtype> output_col_buffer_;
  int output_col_offset_;
  
  int group_;
  
  int weight_offset_;
  int num_;
  int input_channels_;
  int input_height_, input_width_;
  int num_output_;
  int height_out_, width_out_;
  int weight_w_,weight_h_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  int input_col_height_;
  int input_col_width_;
  int M_,N_,K_;
  
};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
