/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/triplet_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/util_img.hpp"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
TripletDataLayer<Dtype>::~TripletDataLayer<Dtype>() {
  this->StopInternalThread();
}


template <typename Dtype>
void TripletDataLayer<Dtype>::PrepareTripletDataParameter() {

  TripletDataParameter* data_param = this->layer_param_.mutable_triplet_data_param();

  if (!data_param->has_image_info()) {
    ImageInfo* image_info = data_param->mutable_image_info();
    image_info->set_height(data_param->height());
    image_info->set_width(data_param->width());
    image_info->set_is_color(data_param->is_color());
  }

  for (int i = 0; i < data_param->affine_image_param_size(); ++i) {
    AffineImageParameter* affine_image_param =
        data_param->mutable_affine_image_param(i);
    if (!affine_image_param->has_image_info()) {
      affine_image_param->mutable_image_info()->CopyFrom(data_param->image_info());
    }
  }
}


template <typename Dtype>
void TripletDataLayer<Dtype>::ImageDataToBlob(shared_ptr<Blob<Dtype> > top, const int item_id, cv::Mat image)
{
	CHECK(top->height() == image.rows);
	CHECK(top->width() == image.cols);
	CHECK(top->channels() == image.channels());
	
	Dtype * topdata = top->mutable_cpu_data();
	int idcount=0;
	const int spsize = image.rows* image.cols *image.channels();
	const int channel = image.channels();

	float meanvalue = this->layer_param_.triplet_data_param().meanvalue();
	float datascale = this->layer_param_.triplet_data_param().datascale();

	if(channel ==3)
	{
		for (int c = 0; c < channel; ++c) 
		{                            
			for (int h = 0; h < image.rows; ++h) {
			    for (int w = 0; w < image.cols; ++w) {

				float v =static_cast<Dtype>(static_cast<unsigned char>(image.at<cv::Vec3b>(h, w)[c]));
				topdata[idcount + item_id * spsize] = (v - meanvalue)*datascale;
				idcount++;
			
			    }
			}
		}
	}
	else
	{
		for (int h = 0; h < image.rows; ++h) {
		    for (int w = 0; w < image.cols; ++w) {

			float v =static_cast<Dtype>(static_cast<char>(image.at<unsigned char>(h, w)));
			topdata[idcount + item_id * spsize] = (v - meanvalue)*datascale;
			idcount++;
		
		    }
		}
	}

}

template <typename Dtype>
void TripletDataLayer<Dtype>::AffineImageAndSetPrefetchData(const pair<string, vector<float> >& sample,const int item_id,LiangjiBatch<Dtype>* batch) {
  cv::Mat img_origin =
      ReadImageToCVMat(imgs_folder_ + "/" + sample.first);
  CHECK(img_origin.data)
    << "Could not read image: "
    << imgs_folder_ + "/" + sample.first;

  vector<float> landmarks(sample.second);
  if (flip_ && (caffe_rng_rand() % 2 == 0)) {
    cv::flip(img_origin, img_origin, 1);

    // TODO
    // 只对左(0)右(1)眼/鼻尖(2)以及左(3)右(4)嘴角变换

    landmarks[0] = img_origin.cols - sample.second[2] - 1;
    landmarks[1] = sample.second[3];
    landmarks[2] = img_origin.cols - sample.second[0] - 1;
    landmarks[3] = sample.second[1];

    landmarks[4] = img_origin.cols - sample.second[4] - 1;
    landmarks[5] = sample.second[5];

    landmarks[6] = img_origin.cols - sample.second[8] - 1;
    landmarks[7] = sample.second[9];
    landmarks[8] = img_origin.cols - sample.second[6] - 1;
    landmarks[9] = sample.second[7];
  }

  for (int top_id = 0; top_id < top_size_ -1; ++top_id) {
    cv::Mat actual_img_orig;
    if (!affine_params_.Get(top_id).image_info().is_color()) {
      cvtColor(img_origin, actual_img_orig, CV_BGR2GRAY);
    } else {
      actual_img_orig = img_origin;
    }
    const AffineImageParameter& image_param = affine_params_.Get(top_id);
    
    GetAffineImage(actual_img_orig, input_imgs_[top_id], landmarks, image_param);
   
    this->ImageDataToBlob(batch->blobs_[top_id], item_id, input_imgs_[top_id]);
 
    /*const int dims = batch->blobs_[top_id].get()->count(1);
    const int offset = item_id * batch->blobs_[top_id].get()->count(1);
    if (mean_values_[top_id].get() != NULL) {
      caffe_sub(dims,
          prefetch_datas_[top_id].get()->mutable_cpu_data() + offset,
          mean_values_[top_id].get()->cpu_data(),
          prefetch_datas_[top_id].get()->mutable_cpu_data() + offset);
    }
    if (affine_params_.Get(top_id).has_scale()) {
      caffe_scal(dims,
          static_cast<Dtype>(affine_params_.Get(top_id).scale()),
          prefetch_datas_[top_id].get()->mutable_cpu_data() + offset);
    }*/
  }
}

template <typename Dtype>
void TripletDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //读入参数, 设置好top 和 prefetch_的size
  PrepareTripletDataParameter();
  const TripletDataParameter& data_param = this->layer_param_.triplet_data_param();
  const TripletDataSubParameter& data_sub_param =
      this->layer_param_.phase() != caffe::TEST ?
          data_param.train_sub_param() : data_param.test_sub_param();

  imgs_folder_ = data_sub_param.imgs_folder();
  key_point_count_ = data_sub_param.key_points_count();
  subjects_per_iter_ = data_sub_param.subjects_per_iter();
  samples_per_subject_ = data_sub_param.samples_per_subject();
  o3_subjects_per_iter_ = data_sub_param.o3_subjects_per_iter();
  o3_samples_per_subject_ = data_sub_param.o3_samples_per_subject();
  is_color_ = data_param.is_color();
  flip_ = data_param.flip();
  LOG(INFO) << "Flip image mode is " << (flip_ ? "on" : "off");

  affine_params_ = data_param.affine_image_param();
  CHECK(top.size() - 1 == affine_params_.size());
  top_size_ = top.size();
  // 算出batchsize
  const int batch_size = subjects_per_iter_ * samples_per_subject_
      + o3_subjects_per_iter_ * o3_samples_per_subject_;
  batch_size_ = batch_size;

  // reshape top and prefetch data
  for(int i=0;i<this->PREFETCH_COUNT;i++)
  {
     this->prefetch_[i].blobs_.clear();
  }
  input_imgs_.clear();
  for (int top_id = 0; top_id < affine_params_.size(); ++top_id) {
    

    for(int i=0;i<this->PREFETCH_COUNT;i++)
    {
       this->prefetch_[i].blobs_.push_back(shared_ptr<Blob<Dtype> >());
    }

    const ImageInfo& image_info = affine_params_.Get(top_id).image_info();
    const int channel = (image_info.is_color() ? 3 : 1);

    for(int i=0;i<this->PREFETCH_COUNT;i++)
    {
       this->prefetch_[i].blobs_[top_id].reset( new Blob<Dtype>(batch_size, channel,image_info.height(), image_info.width()));
    }

    input_imgs_.push_back(cv::Mat(image_info.height(), image_info.width(),
        image_info.is_color() ? CV_8UC3 : CV_8UC1));
    top[top_id]->Reshape(this->prefetch_[0].blobs_[top_id].get()->shape());

    LOG(INFO) << "output data #" << top_id
        << " size: " << top[top_id]->num() << ","
        << top[top_id]->channels() << "," << top[top_id]->height() << ","
        << top[top_id]->width();
  }

  // label
  vector<int> label_shape(1, batch_size);
  for(int i=0;i<this->PREFETCH_COUNT;i++)
  {
       this->prefetch_[i].blobs_.push_back(shared_ptr<Blob<Dtype> >());
       this->prefetch_[i].blobs_[this->prefetch_[i].blobs_.size()-1].reset( new Blob<Dtype>(batch_size, 1,1,1));
  }
  top[top_size_ -1 ]->Reshape(batch_size,1,1,1);
  LOG(INFO) << "output label size: "<< this->prefetch_[0].blobs_[this->prefetch_[0].blobs_.size()-1]->num()<<","
				<< this->prefetch_[0].blobs_[this->prefetch_[0].blobs_.size()-1]->channels()<<","
				<< this->prefetch_[0].blobs_[this->prefetch_[0].blobs_.size()-1]->height()<<","
				<< this->prefetch_[0].blobs_[this->prefetch_[0].blobs_.size()-1]->width();


  // get mean values
  /*mean_values_.clear();
  mean_values_.resize(affine_params_.size());
  for (int top_id = 0; top_id < affine_params_.size(); ++top_id) {
    if (affine_params_.Get(top_id).has_mean_file()) {
      BlobProto mean_blob;
      ReadProtoFromBinaryFileOrDie(affine_params_.Get(top_id).mean_file(),
          &mean_blob);

      mean_values_[top_id].reset(new Blob<Dtype>());
      mean_values_[top_id]->FromProto(mean_blob, true);

      const ImageInfo& image_info = affine_params_.Get(top_id).image_info();
      CHECK_EQ(mean_values_[top_id]->count(),
          (image_info.is_color() ? 3 : 1) * image_info.height() * image_info.width());
    }
  }*/

  if (data_sub_param.has_source_filename() && data_sub_param.has_source_landmark()) {
    LOG(INFO) << "Reading annotations from " << data_sub_param.source_filename()
        << " and " << data_sub_param.source_landmark();
    ReadAnnotations(data_sub_param.source_filename(), data_sub_param.source_landmark(),
        samples_, key_point_count_);
  } else {
    LOG(INFO) << "Reading annotations from " << data_sub_param.source();
    ReadAnnotations(data_sub_param.source(), samples_, key_point_count_);
  }
  // 进行分类
  map<string, int> class2ind;
  map<string, int>::iterator class2ind_iter;
  samples_per_class_.clear();
  for (int i = 0; i < samples_.size(); ++i) {
    const string& filename = samples_[i].first;
    const int split_ind = filename.find_last_of('/');
    CHECK_NE(split_ind, string::npos)
      << filename << " has no class information";

    const string classname = filename.substr(0, split_ind);
    class2ind_iter = class2ind.find(classname);
    if (class2ind_iter == class2ind.end()) {
      const int classind = class2ind.size();
      class2ind_iter = class2ind.insert(class2ind_iter,
          make_pair(classname, classind));
      samples_per_class_.push_back(vector<int>());
    }
    samples_per_class_[class2ind_iter->second].push_back(i);
  }
  LOG(INFO) << "A total of " << samples_.size() << " images.";
  LOG(INFO) << "A total of " << samples_per_class_.size() << " classes.";
  prefetch_inds_.resize(batch_size);

}

template <typename Dtype>
void TripletDataLayer<Dtype>::ShuffleImages() {
  ;
}

// This function is called on prefetch thread
template <typename Dtype>
void TripletDataLayer<Dtype>::load_batch(LiangjiBatch<Dtype>* batch) {
 //LOG(INFO)<<"in load batch ***********";
    CPUTimer batch_timer;
  batch_timer.Start();

  int item_id = 0;
  Dtype* labels = batch->blobs_[top_size_-1]->mutable_cpu_data();
  const int batch_size = batch->blobs_[0]->num();
  CHECK(batch_size == batch_size_);
  //LOG(INFO)<<"before get data in load batch**********";
  int subject_i = 0;
  // subjects_per_iter_ * samples_per_subject_
  while (subject_i < subjects_per_iter_) {
   //LOG(INFO)<<subject_i<<","<<subjects_per_iter_;
    const int cur_subject =
            caffe_rng_rand() % samples_per_class_.size();
    vector<int>& samples_ind = samples_per_class_[cur_subject];
    if (samples_ind.size() == 0) {
      continue;
    }

    int remain_size = samples_per_subject_;
    if (remain_size > 0) {
      shuffle(samples_ind.begin(), samples_ind.end(), caffe_rng());
      for (int sample_i = 0;
          sample_i < remain_size && sample_i < samples_ind.size();
          ++sample_i, ++item_id) {

        const int ind = samples_ind[sample_i];
   //     LOG(INFO)<<"before affine image ";
        AffineImageAndSetPrefetchData(samples_[ind], item_id,batch);
     //   LOG(INFO)<<"end affine image";
        labels[item_id] = static_cast<Dtype>(cur_subject);

        prefetch_inds_[item_id] = ind;
      }
    }
    ++subject_i;
  }
 //LOG(INFO)<<"mid get data in load batch**********";  
  // o3_subjects_per_iter_ * o3_samples_per_subject_
  for (; item_id < batch_size; ++subject_i) {
    const int cur_subject = caffe_rng_rand() % samples_per_class_.size();
    vector<int>& samples_ind = samples_per_class_[cur_subject];
    if (samples_ind.size() == 0) continue;

    for (int i = 0; i < o3_samples_per_subject_ && item_id < batch_size;
        ++i, ++item_id) {
      const int ind = samples_ind[caffe_rng_rand() % samples_ind.size()];
      AffineImageAndSetPrefetchData(samples_[ind], item_id,batch);
      labels[item_id] = static_cast<Dtype>(cur_subject);

      prefetch_inds_[item_id] = ind;
    }
  }
 // LOG(INFO)<<"after get data in load batch**********";  
 // LOG(INFO)<<"top blobsize++++++++++++++ :"<<batch->blobs_.size();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
 // LOG(INFO)<<"out loadbacth ****************";
}

INSTANTIATE_CLASS(TripletDataLayer);
REGISTER_LAYER_CLASS(TripletData);

}  // namespace caffe
#endif  // USE_OPENCV
