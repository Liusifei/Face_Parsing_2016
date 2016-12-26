/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_WEAKGATELSTM_LAYER_HPP_
#define CAFFE_WEAKGATELSTM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class WeakGateLstmLayer : public Layer<Dtype> {/// add by liangji, 20150531
public:
  explicit WeakGateLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WeakGateLstm"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);

  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_, width_;
  int M_;
  int K_x_;
  int K_h_;
  int N_;
  int T_;
  int x_col_count_;
  int h_col_count_;
  int col_length_;

  //Blob<Dtype> x_col_buffer_;
  //Blob<Dtype> h_col_buffer_;
  
  Blob<Dtype> bias_multiplier_;
  
  Blob<Dtype> C_buffer_;
  Blob<Dtype> L_buffer_;
  //Blob<Dtype> P_buffer_;
  Blob<Dtype> G_buffer_;
  
  Blob<Dtype> H_buffer_;
  Blob<Dtype> X_buffer_;
  
  Blob<Dtype> GL_buffer_;
  Blob<Dtype> Trans_buffer_;
  Blob<Dtype> Ct_active_buffer_;
  Blob<Dtype> identical_multiplier_;
  
  bool horizontal_;
  bool reverse_;
  float restrict_w_;
};


}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
