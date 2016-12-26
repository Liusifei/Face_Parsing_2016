/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_TRIPLET_DATA_LAYER_HPP_
#define CAFFE_TRIPLET_DATA_LAYER_HPP_

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
class TripletDataLayer : public BasePrefetchingArbitraryDataLayer<Dtype> {
 public:
  explicit TripletDataLayer(const LayerParameter& param)
      : BasePrefetchingArbitraryDataLayer<Dtype>(param) {}
  virtual ~TripletDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 
  inline const vector<int>& prefetch_inds() const {
    return prefetch_inds_;
  }

  inline const vector<pair<string, vector<float> > >& samples() const {
    return samples_;
  }

  inline const vector<shared_ptr<Blob<Dtype> > >& mean_values() const {
    return mean_values_;
  }

  void ImageDataToBlob(shared_ptr<Blob<Dtype> > top, const int item_id, cv::Mat image);

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(LiangjiBatch<Dtype>* batch);

  void PrepareTripletDataParameter();

  void AffineImageAndSetPrefetchData(const pair<string, vector<float> >& sample, const int item_id,LiangjiBatch<Dtype>* batch);

   string imgs_folder_;
  int key_point_count_;
  int subjects_per_iter_;
  int samples_per_subject_;
  int o3_subjects_per_iter_;
  int o3_samples_per_subject_;
  bool is_color_;
  bool flip_;

  //vector<shared_ptr<Blob<Dtype> > > prefetch_datas_;
  vector<int> prefetch_inds_;
  vector<cv::Mat> input_imgs_;

  google::protobuf::RepeatedPtrField<AffineImageParameter> affine_params_;


  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
  // 所有的样本
  vector<pair<string, vector<float> > > samples_;
  vector<vector<int> > samples_per_class_;
  vector<shared_ptr<Blob<Dtype> > > mean_values_;
  int top_size_;
  int batch_size_;

};


}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_HPP_
