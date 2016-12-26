/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_SEGMENT_DATA_LAYER_HPP_
#define CAFFE_SEGMENT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class SegmentDataLayer : public BasePrefetchingArbitraryDataLayer<Dtype> {
 public:
  explicit SegmentDataLayer(const LayerParameter& param)
      : BasePrefetchingArbitraryDataLayer<Dtype>(param) {}
  virtual ~SegmentDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 
 
  void ImageDataToBlob(shared_ptr<Blob<Dtype> > top, const int item_id, cv::Mat image, vector<float> mean, float datascale);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(LiangjiBatch<Dtype>* batch);

  

  void GetDistribParam();
  void GetImageTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo);
  void GetNUMTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo);
  void GetSparsePointTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo);
  void GetDensePointTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo);
  void GetHeatmapTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo);
  void GetLaneTop(string src,shared_ptr<Blob<Dtype> > top,int item_id,TopInfo topinfo);

  vector<string> lines_;
  int lines_id_;
  int top_size_;
  int batch_size_;
  
  float distrib_angle_;
  float distrib_scale_;
  float distrib_x_;
  float distrib_y_;
  bool use_distrib_;
  bool use_flip_;
  SegmentDataParameter param_;
  string data_folder_;
  //cv::Mat trans_M_;

};


}  // namespace caffe

#endif  // CAFFE_SEGMENT_DATA_LAYER_HPP_
