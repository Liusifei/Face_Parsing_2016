/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_MAPMETRICLOSS_LAYER_HPP_
#define CAFFE_MAPMETRICLOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MapMetricLossLayer : public LossLayer<Dtype> {
 public:
  explicit MapMetricLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "MapMetricLoss"; }
  /**
   * Unlike most loss layers, in the MapMetricLoss we can backpropagate
   * to the first two inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc MapMetricLoss
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Dtype cmpBestThreshold(std::vector<Dtype> simValue,std::vector<Dtype> disValue);
  
  struct Coordinate
  {
     int num;
     int h;
     int w;
  };
  
  int height_;
  int width_;
  int class_num_ ;
  //std::vector < std::vector< std::pair<int, int> > > cls_coord_;
  std::vector < std::vector< Coordinate > > cls_coord_;
  std::vector < int > cls_num_;
  std::vector < Dtype > cls_selectnum_;
  
  Dtype sim_ratio_ ;
  Dtype dis_ratio_ ;
  
  Dtype dis_margin_ ;
  Dtype sim_margin_ ;
  
  std::vector< std::pair<Coordinate, Coordinate > >  select_sim_sample_;
  std::vector< std::pair<Coordinate, Coordinate > >  select_dis_sample_;
  Blob<Dtype> diff_;
  Blob<Dtype> sim_diff_;
  Blob<Dtype> dis_diff_;
  Blob<Dtype> simdiff_sq_;
  Blob<Dtype> disdiff_sq_;
  
};

}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
