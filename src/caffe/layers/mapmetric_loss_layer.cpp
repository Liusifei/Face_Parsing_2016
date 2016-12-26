/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/mapmetricloss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MapMetricLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    CHECK_EQ(bottom[1]->channels(), 1)<<"label map need to be 1 channel";
    CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    CHECK_EQ(bottom[0]->height(), bottom[1]->height());
    CHECK_EQ(bottom[0]->width(), bottom[1]->width());
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    sim_ratio_ = this->layer_param_.mapmetric_loss_param().sim_ratio();
    dis_ratio_ = this->layer_param_.mapmetric_loss_param().dis_ratio();
    sim_margin_ = this->layer_param_.mapmetric_loss_param().sim_margin();
    dis_margin_ = this->layer_param_.mapmetric_loss_param().dis_margin();
    //class_num_ = this->layer_param_.mapmetric_loss_param().class_num();
    class_num_ = bottom[0]->channels();
    CHECK(sim_ratio_>0);
    CHECK(dis_ratio_>0);
   
}

template <typename Dtype>
void MapMetricLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    const Dtype * bottom_data = bottom[0]->cpu_data();
    const Dtype * label = bottom[1]->cpu_data();
    const int dim = bottom[0]->count() / bottom[0]->num();
    const int spdim = bottom[1]->height() * bottom[1]->width();
    cls_coord_.resize(class_num_);
    cls_num_.resize(class_num_,0);
    cls_selectnum_.resize(class_num_,0);
    
    for(int n=0;n<bottom[1]->num();n++)
    {
        for(int h=0;h<bottom[1]->height();h++)
        {
            for(int w=0;w<bottom[1]->width();w++)
            {
                int lb = (int)(label[n*spdim + h*bottom[1]->width()+ w]);
                
                //cls_coord_[lb].push_back(std::make_pair(h, w));
                Coordinate co = {n,h,w};
                cls_coord_[lb].push_back(co);
                cls_num_[lb]+=1;
            }
        }
    }
    
    for(int cls=0;cls<class_num_;cls++)
    {
        cls_selectnum_[cls] = sim_ratio_ * (Dtype)(cls_num_[cls]);
    }
   
    select_sim_sample_.clear();
    select_dis_sample_.clear();
    int id1,id2;
    int other_cls;
    std::vector< int > non_zero_other_cls;
    for(int cls=0;cls<class_num_;cls++)
    {
        for(int p=0;p<cls_selectnum_[cls];p++)
        {
            id1 = caffe::caffe_rng_rand()%cls_num_[cls];
            id2 = caffe::caffe_rng_rand()%cls_num_[cls];
            //select_sim_sample_.push_back(std::make_pair(std::make_pair(cls_coord_[cls][id1].first,cls_coord_[cls][id1].second),
            //                                        std::make_pair(cls_coord_[cls][id2].first,cls_coord_[cls][id2].second)));
            select_sim_sample_.push_back(std::make_pair(cls_coord_[cls][id1],
                                                    cls_coord_[cls][id2]));              
        }
        non_zero_other_cls.clear();
        for(int c=0;c<class_num_;c++)
        {   
            if(c == cls)
                continue;
            if(cls_num_[c]>0)
            {
                non_zero_other_cls.push_back(c);
            }
        }
        if(non_zero_other_cls.size()<1)
            continue;
            
        for(int n=0; n<cls_selectnum_[cls]*dis_ratio_; n++)
        {
            //get dis sim pair
            id1 = caffe::caffe_rng_rand()%cls_num_[cls];
            other_cls = non_zero_other_cls[caffe::caffe_rng_rand()%non_zero_other_cls.size()];    
            id2 = caffe::caffe_rng_rand()%cls_num_[other_cls];   
            //select_dis_sample_.push_back(std::make_pair(std::make_pair(cls_coord_[cls][id1].first,cls_coord_[cls][id1].second),
            //                                        std::make_pair(cls_coord_[other_cls][id2].first,cls_coord_[other_cls][id2].second)));      
            select_dis_sample_.push_back(std::make_pair(cls_coord_[cls][id1],
                                                    cls_coord_[other_cls][id2]));      
        
        }
    }
    CHECK(select_sim_sample_.size() >0);
    sim_diff_.Reshape(select_sim_sample_.size(),class_num_,1,1);
    dis_diff_.Reshape(select_dis_sample_.size(),class_num_,1,1);
    simdiff_sq_.Reshape(select_sim_sample_.size(),1,1,1);
    disdiff_sq_.Reshape(select_dis_sample_.size(),1,1,1);
    Dtype * sim_diff = sim_diff_.mutable_cpu_data();
    Dtype * dis_diff = dis_diff_.mutable_cpu_data();
    Dtype * simdiff_sq = simdiff_sq_.mutable_cpu_data();
    Dtype * disdiff_sq = disdiff_sq_.mutable_cpu_data();
    int n1,n2,h1,w1,h2,w2;
    Dtype loss(0.0);
    
    switch (this->layer_param_.mapmetric_loss_param().losstype()) {
    case MapMetricLossParameter_LossType_contrastive:
        for(int i=0; i<select_sim_sample_.size(); i++)
        {
            n1 = select_sim_sample_[i].first.num;
            h1 = select_sim_sample_[i].first.h;
            w1 = select_sim_sample_[i].first.w;
            n2 = select_sim_sample_[i].second.num;
            h2 = select_sim_sample_[i].second.h;
            w2 = select_sim_sample_[i].second.w;
            for(int j=0; j < class_num_; j++)
            {
                sim_diff[i*class_num_+j] = bottom_data[n1*dim + j* spdim + h1*width_ + w1]
                                                                - bottom_data[n2*dim + +j*spdim + h2*width_ + w2];
            }
            /*
            caffe_sub(
                      class_num_,
                      bottom[0]->cpu_data()+(n1*dim + h1*width_ + w1),  // a
                      bottom[0]->cpu_data()+(n2*dim + h2*width_ + w2),  // b
                      sim_diff_.mutable_cpu_data()+i*class_num_);  // a_i-b_i
                      */
            simdiff_sq[i] = caffe_cpu_dot(class_num_,sim_diff+ i*class_num_, sim_diff+ i*class_num_);
            loss += simdiff_sq[i];                     
        }
        for(int i=0; i<select_dis_sample_.size(); i++)
        {
            n1 = select_dis_sample_[i].first.num;
            h1 = select_dis_sample_[i].first.h;
            w1 = select_dis_sample_[i].first.w;
            n2 = select_dis_sample_[i].second.num;
            h2 = select_dis_sample_[i].second.h;
            w2 = select_dis_sample_[i].second.w;
            for(int j=0; j < class_num_; j++)
            {
                dis_diff[i*class_num_+j] = bottom_data[n1*dim + j* spdim + h1*width_ + w1]
                                                                - bottom_data[n2*dim + +j*spdim + h2*width_ + w2];
            }
            /*
            caffe_sub(
                      class_num_,
                      bottom[0]->cpu_data()+(n1*dim + h1*width_ + w1),  // a
                      bottom[0]->cpu_data()+(n2*dim + h2*width_ + w2),  // b
                      dis_diff_.mutable_cpu_data()+i*class_num_);  // a_i-b_i  
*/                      
            disdiff_sq[i] = caffe_cpu_dot(class_num_,dis_diff+ i*class_num_, 
                                                               dis_diff+ i*class_num_);
            Dtype dist = std::max(dis_margin_ - sqrt(disdiff_sq[i]), 0.0);
            loss += dist*dist;
        }
        
        break;
    case MapMetricLossParameter_LossType_weakcontrastive:
        break;
    default:
      LOG(FATAL) << "Unknown loss type";
    }  
    
}

template <typename Dtype>
void MapMetricLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    

    caffe_set(bottom[0]->count(),Dtype(0.0),bottom[0]->mutable_cpu_diff());
    
    const int dim = bottom[0]->count() / bottom[0]->num();
    const int spdim = bottom[1]->height() * bottom[1]->width();
    Dtype alpha =  top[0]->cpu_diff()[0] /(select_sim_sample_.size()+select_dis_sample_.size());
        //const Dtype * bottom_data = bottom[0]->cpu_data();
        const Dtype * sim_diff = sim_diff_.cpu_data();
        const Dtype * dis_diff = dis_diff_.cpu_data();
        Dtype * bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype * disdiff_sq = disdiff_sq_.cpu_data();
    
    switch (this->layer_param_.mapmetric_loss_param().losstype()) {
    case MapMetricLossParameter_LossType_contrastive:
        
        int n1,n2,h1,w1,h2,w2;
        for(int i=0; i<select_sim_sample_.size(); i++)
        {
            n1 = select_sim_sample_[i].first.num;
            h1 = select_sim_sample_[i].first.h;
            w1 = select_sim_sample_[i].first.w;
            n2 = select_sim_sample_[i].second.num;
            h2 = select_sim_sample_[i].second.h;
            w2 = select_sim_sample_[i].second.w;
            
            for(int j=0; j < class_num_; j++)
            {
                bottom_diff[n1*dim + j* spdim + h1*width_ + w1] += alpha * sim_diff[i*class_num_+j];
                bottom_diff[n2*dim + j* spdim + h2*width_ + w2] += -alpha * sim_diff[i*class_num_+j];
            }           
        }
        for(int i=0; i<select_dis_sample_.size(); i++)
        {
            n1 = select_dis_sample_[i].first.num;
            h1 = select_dis_sample_[i].first.h;
            w1 = select_dis_sample_[i].first.w;
            n2 = select_dis_sample_[i].second.num;
            h2 = select_dis_sample_[i].second.h;
            w2 = select_dis_sample_[i].second.w;
            
            Dtype mdist(0.0);
            Dtype beta(0.0);
            Dtype dist = sqrt(disdiff_sq[i]);
            mdist = dis_margin_ - dist;
            beta = alpha * mdist / (dist + Dtype(1e-4));
            if(mdist > 0)
            {
                for(int j=0; j < class_num_; j++)
                {
                    bottom_diff[n1*dim + j* spdim + h1*width_ + w1] += -beta * dis_diff[i*class_num_+j];
                    bottom_diff[n2*dim + j* spdim + h2*width_ + w2] += beta * dis_diff[i*class_num_+j];
                }     
            }
        }
        break;
    case MapMetricLossParameter_LossType_weakcontrastive:
        break;
    default:
      LOG(FATAL) << "Unknown loss type";
    }  
}

template <typename Dtype>
Dtype MapMetricLossLayer<Dtype>::cmpBestThreshold(std::vector<Dtype> simValue,std::vector<Dtype> disValue){
  std::sort(disValue.begin(),disValue.end());
  std::sort(simValue.begin(),simValue.end());
  Dtype bestThreshold=disValue[0]-Dtype(0.000000001);
  int negIdx=simValue.size()-1;
  while(negIdx>=0&&simValue[negIdx]>=bestThreshold)
    negIdx--;
  int minErroCount=simValue.size()-negIdx-1;
  for(int i=0;i<disValue.size();i++){
    while(negIdx<(static_cast<int>(simValue.size())-1)&&simValue[negIdx+1]<(disValue[i]+Dtype(0.000000001)))
      negIdx++;
    int errc=simValue.size()-negIdx+i;
    if(errc<minErroCount){
      minErroCount=errc;
      bestThreshold=disValue[i]+Dtype(0.000000001);
    }
  }
  return bestThreshold;
}

#ifdef CPU_ONLY
STUB_GPU(MapMetricLossLayer);
#endif

INSTANTIATE_CLASS(MapMetricLossLayer);
REGISTER_LAYER_CLASS(MapMetricLoss);

}  // namespace caffe
