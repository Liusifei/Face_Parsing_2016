#ifndef CAFFE_CONCAT_LAYER_HPP_
#define CAFFE_CONCAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class Point4D {
public:
	int n,c,y,x;
	Point4D ( int n_,int c_,int y_, int x_){
		n = n_;
		c = c_;
		y = y_;
		x = x_;
	}
};
template <typename Dtype>
class HardSampleLayer : public Layer<Dtype> {
 public:
  explicit HardSampleLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HardSample"; }
 
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  //do not auto set top
  virtual inline bool AutoTopBlobs() const { return false; }
  
 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  std::vector<int> get_permutation(int start, int end, bool random);
  static bool comparePointScore(const std::pair< Point4D ,Dtype>& c1,
	    const std::pair< Point4D ,Dtype>& c2);
  
  void get_all_pos_neg_instances(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
  void set_mask(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
  
  Blob<Dtype> mask_;
  std::vector < std::vector <std::pair< Point4D ,Dtype> > > negative_points_;
  std::vector < std::vector <std::pair< Point4D ,Dtype> > > positive_points_;
  int pos_margin_;
  int neg_margin_;
  
};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
