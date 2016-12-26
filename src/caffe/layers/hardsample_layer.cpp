/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>
#include <cfloat>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/hardsample_layer.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void HardSampleLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
   //CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK(bottom[0]->channels() == bottom[1]->channels())<< "Inputs must have the same dimension.";
   
   if(bottom.size()>2)
   {
        CHECK_EQ(bottom[0]->num(), bottom[2]->num())<< "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->height(), bottom[2]->height())<< "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->width(), bottom[2]->width())<< "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->channels() , bottom[2]->channels() )<< "Inputs must have the same dimension.";
   }
  
   
   top[0]->ReshapeLike(*bottom[0]);
   mask_.ReshapeLike(*bottom[0]);
   for(int i = 0; i< bottom[0]->channels(); i++)
   {
      vector <std::pair< Point4D ,Dtype> > list_neg;
      this->negative_points_.push_back(list_neg);
      vector <std::pair< Point4D ,Dtype> > list_pos;
      this->positive_points_.push_back(list_pos);
   }
   neg_margin_ = this->layer_param_.hardsample_param().neg_margin();
   pos_margin_ = this->layer_param_.hardsample_param().pos_margin();
  
   //use_balancesample_ = this->layer_param_.hardsample_param().use_balancesample();
}

template <typename Dtype>
void HardSampleLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
   //CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK(bottom[0]->channels() == bottom[1]->channels() )<< "Inputs must have the same dimension.";
   if(bottom.size()>2)
   {
        CHECK_EQ(bottom[0]->num(), bottom[2]->num())<< "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->height(), bottom[2]->height())<< "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->width(), bottom[2]->width())<< "Inputs must have the same dimension.";
        CHECK_EQ(bottom[0]->channels() , bottom[2]->channels() )<< "Inputs must have the same dimension.";
   }
   
   top[0]->ReshapeLike(*bottom[0]);
   mask_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HardSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    if(bottom.size()>2)
    {
        caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(),bottom[2]->cpu_data(), top[0]->mutable_cpu_data());
    }
    else
    {
        caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
    }
    
}

template <typename Dtype>
void HardSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
    if (propagate_down[0]) {
        if(this->layer_param_.hardsample_param().use_hardsample())
        {
            set_mask(bottom,top);
            caffe_mul(bottom[0]->count(), top[0]->cpu_diff(),mask_.cpu_data(), bottom[0]->mutable_cpu_diff());
        }
        else
        {
            if(bottom.size()>2)
            {
                caffe_mul(top[0]->count(), top[0]->cpu_diff(),bottom[2]->cpu_data(), bottom[0]->mutable_cpu_diff());
            }
            else
            {
                caffe_copy(top[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
            }
        }
    }
}
template <typename Dtype>
void HardSampleLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
{
    const int count = bottom[0]->count();
 //   int num = bottom[0]->num();
    //int dim = bottom[0]->count() / num;
//	int spatial_dim = bottom[0]->height() * bottom[0]->width();
//	int channels = bottom[0]->channels();
    //const Dtype * label_data = bottom[1]->cpu_data();
    //Dtype * top_diff = top[0]->mutable_cpu_diff();
    Dtype * mask_data = mask_.mutable_cpu_data();
    
    //clear mask
    caffe_set(count,Dtype(0.0),mask_data);
    
    //get pos and neg points
    get_all_pos_neg_instances(bottom,top);
    
    
    float pos_use_ratio = this->layer_param_.hardsample_param().pos_use_ratio();
    float neg_use_ratio = this->layer_param_.hardsample_param().neg_use_ratio();
    float pos_hard_ratio = this->layer_param_.hardsample_param().pos_hard_ratio();
    float neg_hard_ratio = this->layer_param_.hardsample_param().neg_hard_ratio();
    float neg_compare_pos_ratio = this->layer_param_.hardsample_param().neg_compare_pos_ratio();
    float pos_ignore_hardest_ratio = this->layer_param_.hardsample_param().pos_ignore_hardest_ratio();
    float neg_ignore_hardest_ratio = this->layer_param_.hardsample_param().neg_ignore_hardest_ratio();
    
    
 //   int min_neg_num = this->layer_param_.hardsample_param().min_neg_num();
 //   int min_pos_num = this->layer_param_.hardsample_param().min_pos_num();
    int max_ignore_pos_hardest_num = this->layer_param_.hardsample_param().max_ignore_pos_hardest_num();
    int max_ignore_neg_hardest_num = this->layer_param_.hardsample_param().max_ignore_neg_hardest_num();
    
    
    // assign mask
	for(int c=0; c< bottom[0]->channels();++c){
		// get negative count according to negative ratio
        int all_pos_count = positive_points_[c].size();
        int all_neg_count = negative_points_[c].size();
        
        int pos_count = all_pos_count * pos_use_ratio;
        int neg_count = all_neg_count * neg_use_ratio;
        int pos_ignore_count = all_pos_count * pos_ignore_hardest_ratio;
        int neg_ignore_count = all_neg_count * neg_ignore_hardest_ratio;
        if(max_ignore_pos_hardest_num > 0)
        {    
            pos_ignore_count = std::min(pos_ignore_count, max_ignore_pos_hardest_num);
            pos_ignore_count = std::min(pos_ignore_count, all_pos_count);
        }
        if(max_ignore_neg_hardest_num > 0)
        {
            neg_ignore_count = std::min(neg_ignore_count, max_ignore_neg_hardest_num);
            neg_ignore_count = std::min(neg_ignore_count, all_neg_count);
        }

        pos_count = std::min(pos_count,all_pos_count - pos_ignore_count);
        neg_count = std::min(neg_count,all_neg_count - neg_ignore_count);
        
        if(neg_compare_pos_ratio > 0 && all_pos_count > 0 && all_neg_count > 0)
        {
            neg_count = int(float(pos_count) * neg_compare_pos_ratio);
            if(neg_count > all_neg_count - neg_ignore_count)
            {
                float ratio =  float(all_neg_count - neg_ignore_count)/ float(neg_count) ;
                CHECK(ratio < 1);
                pos_count = int(float(pos_count) * ratio);
                neg_count = all_neg_count - neg_ignore_count;
            }
            //neg_count = std::min(neg_count,all_neg_count);
        }
        //pos_count = std::max(min_pos_num,pos_count);
        //neg_count = std::max(min_neg_num,neg_count);
        
        
        
        CHECK(pos_count >= 0&& pos_count <= all_pos_count);
        CHECK(neg_count >= 0&& neg_count <= all_neg_count);
        
        pos_ignore_count = std::min(pos_ignore_count,all_pos_count - pos_count);
        neg_ignore_count = std::min(neg_ignore_count,all_neg_count - neg_count);
        
        int hard_pos_count = pos_count * pos_hard_ratio;
        int hard_neg_count = neg_count * neg_hard_ratio;
        hard_pos_count = std::min(std::max(hard_pos_count,0),pos_count);
        hard_neg_count = std::min(std::max(hard_neg_count,0),neg_count);
        
        std::stable_sort(negative_points_.at(c).begin(), negative_points_.at(c).end(),HardSampleLayer<Dtype>::comparePointScore);
        std::stable_sort(positive_points_.at(c).begin(), positive_points_.at(c).end(),HardSampleLayer<Dtype>::comparePointScore);


		// sample the hard_pos_count hard pos samples  from the top  pos_count hard pos candidates
		int pos_hard_count_sampling_range = (std::min((unsigned int)(pos_count),(unsigned int)(positive_points_.at(c).size() - pos_ignore_count)));
		pos_hard_count_sampling_range = std::max(0,pos_hard_count_sampling_range);
		vector<int> hard_positive_random_ids = get_permutation(pos_ignore_count,pos_ignore_count + pos_hard_count_sampling_range, true);
		for(int hard_pos_id = 0 ; hard_pos_id < hard_pos_count; ++hard_pos_id){
			int id = hard_positive_random_ids[hard_pos_id];
			Point4D point =positive_points_.at(c)[id].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_.offset(point.n,point.c,point.y, point.x)]= 1;
		}

		// sample the remaining positive samples from all the positive candidates
		hard_positive_random_ids = get_permutation(pos_ignore_count, positive_points_.at(c).size(), true);
		for(int pos_id = 0; pos_id < pos_count-hard_pos_count; pos_id++){
			int id = hard_positive_random_ids[pos_id];
			Point4D point =positive_points_.at(c)[id].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_.offset(point.n,point.c,point.y, point.x)]= 1;
		}
        
        
        // sample the hard_neg_count hard neg samples  from the top  neg_count hard neg candidates
		int neg_hard_count_sampling_range = (std::min((unsigned int)(neg_count),(unsigned int)(negative_points_.at(c).size() - neg_ignore_count)));
		neg_hard_count_sampling_range = std::max(0,neg_hard_count_sampling_range);
		vector<int> hard_negative_random_ids = get_permutation(neg_ignore_count,neg_ignore_count + neg_hard_count_sampling_range, true);
		for(int hard_neg_id = 0 ; hard_neg_id < hard_neg_count; ++hard_neg_id){
			int id = hard_negative_random_ids[hard_neg_id];
			Point4D point =negative_points_.at(c)[id].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_.offset(point.n,point.c,point.y, point.x)]= 1;
		}

		// sample the remaining negative samples from all the negative candidates
		hard_negative_random_ids = get_permutation(neg_ignore_count, negative_points_.at(c).size(), true);
		for(int neg_id = 0; neg_id < neg_count-hard_neg_count; neg_id++){
			int id = hard_negative_random_ids[neg_id];
			Point4D point =negative_points_.at(c)[id].first;
			//LOG(INFO)<<"score: "<<negative_points_[neg_id].second<<" point: "<<point.n<<"  "<<point.c<<"  "<<point.y <<"  "<<point.x;
			mask_data[mask_.offset(point.n,point.c,point.y, point.x)]= 1;
		}
  
		if(this->layer_param_.hardsample_param().print_info())
			LOG(INFO)<<"in channel "<<c<<" , neg_points size:" <<negative_points_.at(c).size()
                        <<" , pos_points size:" <<positive_points_.at(c).size()
                        <<" neg_count: "<<neg_count <<"  pos_count "<<pos_count
						<<"  total size: "<<neg_count+pos_count  <<" hard_negative: "<<hard_neg_count 
                        <<" hard_positive: "<<hard_pos_count  ;
	}

}


template <typename Dtype>
void HardSampleLayer<Dtype>::get_all_pos_neg_instances(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
{
    int num = bottom[0]->num();
    //int dim = bottom[0]->count() / num;
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	int channels = bottom[0]->channels();
    int height = bottom[0]->height();
    int width = bottom[0]->width();
    

	const Dtype * label_data = bottom[1]->cpu_data();
	//const Dtype * predicted_data = bottom[0]->cpu_data();
    Dtype * top_diff = top[0]->mutable_cpu_diff();
	Dtype * ignore_labels = bottom.size() > 2? bottom[2]->mutable_cpu_data() : NULL;


    for(int i=0;i<channels;++i){
		this->negative_points_.at(i).clear();
        this->positive_points_.at(i).clear();
	}
    
	for(int n_id = 0; n_id < num; ++n_id){

		//int cur_pos = 0;
		for(int c_id = 0 ; c_id < channels; ++ c_id){
			const Dtype* label_base_ptr = &(label_data[bottom[1]->offset(n_id,c_id)]);
			const Dtype* diff_values_ptr = &(top_diff[top[0]->offset(n_id, c_id)]);
            //const Dtype* predicted_values_ptr = &(predicted_data[bottom[0]->offset(n_id, c_id)]);
			Dtype* ignore_label_ptr = NULL;

			if(ignore_labels != NULL)
				 ignore_label_ptr = &(ignore_labels[bottom[2]->offset(n_id, c_id)]);
		
			for(int cur_off = 0; cur_off < spatial_dim; ++cur_off){
				
				if(ignore_label_ptr != NULL && static_cast<int>(ignore_label_ptr[cur_off]) != 1)
					continue;
                    
				//  for positive instance
				if( (label_base_ptr[cur_off]) != Dtype(0)){
					bool find_negative = false;
					for(int margin_row = pos_margin_*-1;margin_row <= pos_margin_; ++margin_row){
						for(int margin_col = pos_margin_*-1; margin_col <= pos_margin_; ++margin_col){
							int row_id = std::min(std::max(cur_off/width+margin_row,0),height-1);
							int col_id = std::min(std::max(cur_off%width+margin_col,0),width-1);
							if(label_base_ptr[row_id*width+col_id] != 1){
								find_negative = true;
								break;
							}
						}
						if(find_negative)
							break;
					}
					if(find_negative == false){
						positive_points_.at(c_id).push_back(std::make_pair(Point4D (n_id,c_id,
							cur_off/width, cur_off%width),std::abs(diff_values_ptr[cur_off])));
					}
				}
				// for negative instances
				else{
					bool find_positive = false;
					for(int margin_row = neg_margin_*-1;margin_row <= neg_margin_; ++margin_row){
						for(int margin_col = neg_margin_*-1; margin_col <= neg_margin_; ++margin_col){
							int row_id = std::min(std::max(cur_off/width+margin_row,0),height-1);
							int col_id = std::min(std::max(cur_off%width+margin_col,0),width-1);
							if(label_base_ptr[row_id*width+col_id] != 0){
								find_positive = true;
								break;
							}
						}
						if(find_positive)
							break;
					}
					if(find_positive == false){
						negative_points_.at(c_id).push_back(std::make_pair(Point4D (n_id,c_id,
							cur_off/width, cur_off%width),std::abs(diff_values_ptr[cur_off])));
					}
				}
			}
		}
	}
}
template <typename Dtype>
bool HardSampleLayer<Dtype>::comparePointScore(const std::pair< Point4D,Dtype>& c1,
    const std::pair< Point4D,Dtype>& c2) {
  return c1.second >= c2.second;
}

template <typename Dtype>
std::vector<int> HardSampleLayer<Dtype>::get_permutation(int start, int end, bool random)
{
	vector<int> res;
	for(int i= start; i < end; ++i)
		res.push_back(i);
	if(random)
		caffe::shuffle(res.begin(),res.end());
	return res;
}

#ifdef CPU_ONLY
STUB_GPU(HardSampleLayer);
#endif

INSTANTIATE_CLASS(HardSampleLayer);
REGISTER_LAYER_CLASS(HardSample);

}  // namespace caffe
