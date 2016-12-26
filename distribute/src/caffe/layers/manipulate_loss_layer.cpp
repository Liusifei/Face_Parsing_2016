/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>
#include <cfloat>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/manipulateloss_layer.hpp"

namespace caffe {

template <typename Dtype>
void ManipulateLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
   //CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK(bottom[0]->channels() == bottom[1]->channels() || bottom[1]->channels()==1 )<< "Inputs must have the same dimension.";
   
   if(bottom.size()==3)
   {
       CHECK_EQ(bottom[0]->num(), bottom[2]->num())<< "Inputs must have the same dimension.";
       CHECK_EQ(bottom[0]->height(), bottom[2]->height())<< "Inputs must have the same dimension.";
       CHECK_EQ(bottom[0]->width(), bottom[2]->width())<< "Inputs must have the same dimension.";
       CHECK(bottom[2]->channels()==1 )<< "mask channel should be 1, here get "<<bottom[2]->channels();
   }
   
   top[0]->ReshapeLike(*bottom[0]);
   diff_.ReshapeLike(*bottom[0]);
   use_balancesample_ = this->layer_param_.manipulate_loss_param().use_balancesample();
}

template <typename Dtype>
void ManipulateLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
   //CHECK_EQ(bottom[0]->count(), bottom[1]->count())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->num(), bottom[1]->num())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->height(), bottom[1]->height())<< "Inputs must have the same dimension.";
   CHECK_EQ(bottom[0]->width(), bottom[1]->width())<< "Inputs must have the same dimension.";
   CHECK(bottom[0]->channels() == bottom[1]->channels() || bottom[1]->channels()==1 )<< "Inputs must have the same dimension.";
   
   
   top[0]->ReshapeLike(*bottom[0]);
   diff_.ReshapeLike(*bottom[0]);
   if(this->layer_param_.manipulate_loss_param().use_unionchannel_balance())
   {
        mask_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
   }
}

template <typename Dtype>
void ManipulateLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void ManipulateLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    caffe_copy(top[0]->count(), top[0]->cpu_diff(), diff_.mutable_cpu_diff());
    
    //const Dtype * prob_data = bottom[0]->cpu_data();
	const Dtype * label_data = bottom[1]->cpu_data();
    Dtype * diff_data = diff_.mutable_cpu_diff();
      
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;
	int spatial_dim = bottom[0]->height() * bottom[0]->width();
	int channels = bottom[0]->channels();
      
    if(use_balancesample_)
    {
      
	  per_class_statistic.resize(channels);
	  per_class_balanceRate.resize(channels);
	  for(int k=0;k<channels;k++)
	  {
		per_class_statistic[k]=0;
		per_class_balanceRate[k]=0;
	  }
	  
	  for (int i = 0; i < num; ++i) 
	  {
		for (int j = 0; j < spatial_dim; j++) 
		{
		  Dtype maxval = -FLT_MAX;
		  int max_id = -1;
		  int lb=0;
            if(bottom[1]->channels()==1)
            {   int idx = i * spatial_dim + j;
                lb = label_data[idx];
            }
            else
            {
                for (int c = 0; c < channels; ++c) 
                {
                    int idx = i * dim + c * spatial_dim + j;
                    if ( label_data[idx] > maxval) {
                      maxval = label_data[idx];
                      max_id = c;
                    }
                }
                lb = max_id;
            }
			per_class_statistic[lb]+=1;
		}
	  }
		Dtype mean_classnum=0;
		int valid_classnum=0;
		  //assume label 0 means background, largest in label map
		  for(int c=1;c<channels;c++)
		  {
			if(per_class_statistic[c]>0)
			{
				mean_classnum += per_class_statistic[c];
				valid_classnum +=1;
			}
		  }
       if(valid_classnum == 0)
		  { 
			 
			LOG(INFO) << this->layer_param().name()<< " do not have any valid class label , all label is 0 in current batch!!";
			mean_classnum = per_class_statistic[0];
			valid_classnum = channels;
			if(valid_classnum ==0)
			{
				LOG(FATAL) << this->layer_param().name()<< "valid_num  channels == 0!!";
			}	
		  }
		  
		  mean_classnum /= valid_classnum;
		  Dtype threshold = mean_classnum * this->layer_param_.manipulate_loss_param().bg_ratio();
		  for(int c=0;c<channels;c++)
		  {
			if(per_class_statistic[c]<= threshold)
			{
				per_class_balanceRate[c]=1;
			}
			else
			{
				per_class_balanceRate[c]= threshold / per_class_statistic[c];
			}
            
            //uniform drop diff
            if(this->layer_param_.manipulate_loss_param().uniform_droprate()<1.0)
            {
                CHECK(this->layer_param_.manipulate_loss_param().uniform_droprate() > 0.0);
                per_class_balanceRate[c] =  per_class_balanceRate[c] * this->layer_param_.manipulate_loss_param().uniform_droprate();
            }
		  }
          
          
	  
      
	      for (int i = 0; i < num; ++i) 
		  {
			for (int j = 0; j < spatial_dim; j++) 
			{
			  Dtype maxval = -FLT_MAX;
			  int max_id = -1;
			  int lb=0;
              if(bottom[1]->channels()==1)
              {   int idx = i * spatial_dim + j;
                lb = label_data[idx];
              }
              else
              {
                  for (int c = 0; c < channels; ++c) 
                  {
                        int idx = i * dim + c * spatial_dim + j;
                        if ( label_data[idx] > maxval) {
                          maxval = label_data[idx];
                          max_id = c;
                        }
                    }
                    lb = max_id;
                }
				 // -------------------- balance ---------------------//
					float rd = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1));
					if ( rd > per_class_balanceRate[lb]) {
					   for (int c = 0; c < channels; ++c) {
						  diff_data[i * dim + c * spatial_dim + j] = 0;
					   }
					}
					// -------------------- balance ---------------------//

			}
		  }
        }//end use balance sample
        if(this->layer_param_.manipulate_loss_param().use_perchannel_balance())
        {
              CHECK(bottom[0]->channels() == bottom[1]->channels())<<"use_channel_balance in manipulate, channel should be same";
              per_class_statistic.resize(2);
              per_class_balanceRate.resize(2);
              

              for(int cur_channel = 0; cur_channel < bottom[1]->channels(); cur_channel++)
              {
                  for(int k=0;k<2;k++)
                  {
                    per_class_statistic[k]=0;
                    per_class_balanceRate[k]=0;
                  }
              
                  for (int i = 0; i < num; ++i) 
                  {
                    for (int j = 0; j < spatial_dim; j++) 
                    {
                        int lb=0;
                        int idx = i * dim + cur_channel*spatial_dim + j;
                        lb = label_data[idx]>0?1:0;
                        per_class_statistic[lb]+=1;
                    }
                  }
                  Dtype mean_classnum=0;
                  mean_classnum = std::min(per_class_statistic[0],per_class_statistic[1]);
                  if(mean_classnum <1)
                  {
                     mean_classnum = std::max(per_class_statistic[0],per_class_statistic[1])/this->layer_param_.manipulate_loss_param().bg_ratio();
                     if(this->layer_param_.manipulate_loss_param().drop_singleclasschannel())
                     {
                         mean_classnum = 0;
                     }
                  }

                  //int valid_classnum=0;
                   /*   
                  for(int c=1;c<2;c++)
                  {
                    if(per_class_statistic[c]>0)
                    {
                        mean_classnum += per_class_statistic[c];
                        valid_classnum +=1;
                    }
                  }*/
                  /*
                   if(valid_classnum == 0)
                   { 
                        LOG(INFO) << this->layer_param().name()<< " do not have any valid class label , all label is 0 in current batch, at channel "<<cur_channel;
                        mean_classnum = per_class_statistic[0];
                        valid_classnum = 2;
                        if(valid_classnum ==0)
                        {
                            LOG(FATAL) << this->layer_param().name()<< "valid_num  channels == 0!!";
                        }	
                   }
                   */
                      
                      //mean_classnum /= valid_classnum;
                      Dtype threshold = mean_classnum * this->layer_param_.manipulate_loss_param().bg_ratio();
                      for(int c=0;c<2;c++)
                      {
                        if(per_class_statistic[c]<= threshold)
                        {
                            per_class_balanceRate[c]=1;
                        }
                        else
                        {
                            per_class_balanceRate[c]= threshold / per_class_statistic[c];
                        }
                        
                        //uniform drop diff
                        if(this->layer_param_.manipulate_loss_param().uniform_droprate()<1.0)
                        {
                            CHECK(this->layer_param_.manipulate_loss_param().uniform_droprate() > 0.0);
                            per_class_balanceRate[c] =  per_class_balanceRate[c] * this->layer_param_.manipulate_loss_param().uniform_droprate();
                        }
                      }
                      
                  
                      int dropnum[2];
                      dropnum[0]=0;
                      dropnum[1]=0;
                      for (int i = 0; i < num; ++i) 
                      {
                        for (int j = 0; j < spatial_dim; j++) 
                        {
                          
                          int lb=0;
                          int idx = i * dim + cur_channel*spatial_dim + j;
                          lb = label_data[idx]>0?1:0;
                                                 
                         // -------------------- balance ---------------------//
                            float rd = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1));
                            if ( rd > per_class_balanceRate[lb]) {
                               
                                diff_data[i * dim + cur_channel * spatial_dim + j] = 0;
                                dropnum[lb]+=1;
                            }
                            // -------------------- balance ---------------------//

                        }
                      }
                      if(this->layer_param_.manipulate_loss_param().print_info())
                      {
                        LOG(INFO)<<"in channel "<<cur_channel<<", label 0 num "
                                 <<per_class_statistic[0]<<", label 1 num "
                                 <<per_class_statistic[1]<<" threshold ="<<threshold
                                 <<", per_class_balanceRate[0]="<<per_class_balanceRate[0]
                                 <<", per_class_balanceRate[1]="<<per_class_balanceRate[1]
                                 <<", lb 0 dropnum = "<<dropnum[0]
                                 <<", lb 1 dropnum = "<<dropnum[1];
                      }
                }
        }//end use pre channel balance sample
        if(this->layer_param_.manipulate_loss_param().use_unionchannel_balance())
        {
              Dtype * mask_data = mask_.mutable_cpu_data();
              caffe_set(mask_.count(),Dtype(1),mask_data);
              
              CHECK(bottom[0]->channels() == bottom[1]->channels())<<"use_channel_balance in manipulate, channel should be same";
              per_class_statistic.resize(2);
              per_class_balanceRate.resize(2);
              
              for(int k=0;k<2;k++)
              {
                per_class_statistic[k]=0;
                per_class_balanceRate[k]=0;
              }
                for(int c = 0;c < bottom[0]->channels() ;c++)
                {
                    caffe_mul(mask_.count(),mask_data,label_data + c*spatial_dim,mask_data);
                }
                per_class_statistic[1] = mask_.asum_data();
                per_class_statistic[0] = mask_.count() - per_class_statistic[1];
                  
                Dtype mean_classnum=0;
                mean_classnum = std::min(per_class_statistic[0],per_class_statistic[1]);
                if(mean_classnum <1)
                {    
                    mean_classnum = std::max(per_class_statistic[0],per_class_statistic[1])/this->layer_param_.manipulate_loss_param().bg_ratio();
                     if(this->layer_param_.manipulate_loss_param().drop_singleclasschannel())
                     {
                         mean_classnum = 0;
                     }
                }
                  
                  //mean_classnum /= valid_classnum;
                  Dtype threshold = mean_classnum * this->layer_param_.manipulate_loss_param().bg_ratio();
                  for(int c=0;c<2;c++)
                  {
                    if(per_class_statistic[c]<= threshold)
                    {
                        per_class_balanceRate[c]=1;
                    }
                    else
                    {
                        per_class_balanceRate[c]= threshold / per_class_statistic[c];
                    }
                  
                    //uniform drop diff
                    if(this->layer_param_.manipulate_loss_param().uniform_droprate()<1.0)
                    {
                        CHECK(this->layer_param_.manipulate_loss_param().uniform_droprate() > 0.0);
                        per_class_balanceRate[c] =  per_class_balanceRate[c] * this->layer_param_.manipulate_loss_param().uniform_droprate();
                    }
                  }
                  
                  
                      int dropnum[2];
                      dropnum[0]=0;
                      dropnum[1]=0;
                      for (int i = 0; i < num; ++i) 
                      {
                        for (int j = 0; j < spatial_dim; j++) 
                        {
                          
                          int lb=0;
                          int idx = i*spatial_dim + j;
                          lb = mask_data[idx]>0?1:0;
                                                 
                         // -------------------- balance ---------------------//
                            float rd = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1));
                            if ( rd > per_class_balanceRate[lb]) {
                                for (int c = 0; c < channels; ++c) {
                                    diff_data[i * dim + c * spatial_dim + j] = 0;
                                    dropnum[lb]+=1;
                                } 
                            }
                            // -------------------- balance ---------------------//

                        }
                      }
                      if(this->layer_param_.manipulate_loss_param().print_info())
                      {
                        LOG(INFO)<<"label 0 num "
                                 <<per_class_statistic[0]<<", label 1 num "
                                 <<per_class_statistic[1]<<" threshold ="<<threshold
                                 <<", per_class_balanceRate[0]="<<per_class_balanceRate[0]
                                 <<", per_class_balanceRate[1]="<<per_class_balanceRate[1]
                                 <<", lb 0 dropnum = "<<dropnum[0]
                                 <<", lb 1 dropnum = "<<dropnum[1];
                      }
                      
           
                
        }//end use union channel balance sample
        std::string ignorevalue = this->layer_param_.manipulate_loss_param().ignore_value();
        if(ignorevalue.size()>0)
        {
            vector<std::string> values = str_split(ignorevalue,"||");
            for(int v=0;v< values.size();v++)
            {
                int value = atoi(values[v].c_str());
                
                for (int i = 0; i < num; ++i) 
                  {
                    for (int j = 0; j < spatial_dim; j++) 
                    {
                      Dtype maxval = -FLT_MAX;
                      int max_id = -1;
                      int lb=0;
                      if(bottom[1]->channels()==1)
                      {   
                        int idx = i * spatial_dim + j;
                        lb = label_data[idx];
                      }
                      else{
                          for (int c = 0; c < channels; ++c) 
                          {
                                int idx = i * dim + c * spatial_dim + j;
                                if ( label_data[idx] > maxval) {
                                  maxval = label_data[idx];
                                  max_id = c;
                                }
                            }
                            lb = max_id;
                        }
                     // -------------------- balance ---------------------//
                        
                        if ( lb == value) {
                           for (int c = 0; c < channels; ++c) {
                              diff_data[i * dim + c * spatial_dim + j] = 0;
                           }
                        }
                        // -------------------- balance ---------------------//

                    }
                  }
                
            }
        }//end ingorevalue
        
        if(this->layer_param_.manipulate_loss_param().use_fullzeroignore())
        {
           // vector<std::string> values = str_split(ignorevalue,"||");
            //for(int v=0;v< values.size();v++)
            //{
                //int value = atoi(values[v].c_str());
                
                for (int i = 0; i < num; ++i) 
                  {
                    for (int j = 0; j < spatial_dim; j++) 
                    {
                      int valid_num = 0;
                      CHECK(bottom[1]->channels()>1 );
                    
                      for (int c = 0; c < channels; ++c) 
                      {
                            int idx = i * dim + c * spatial_dim + j;
                            if ( label_data[idx] > 0) {
                                valid_num +=1;
                            }
                        }
                       
                      
                     // -------------------- balance ---------------------//
                        
                        if ( valid_num == 0) {
                           for (int c = 0; c < channels; ++c) {
                              diff_data[i * dim + c * spatial_dim + j] = 0;
                           }
                        }
                        // -------------------- balance ---------------------//

                    }
                  }
                
           // }
        }//end ingorevalue
        if(bottom.size()==3)//use mask to clear diff
        {
              const Dtype * mask_data = bottom[2]->cpu_data();
              for (int i = 0; i < num; ++i) 
              {
                for (int j = 0; j < spatial_dim; j++) 
                {  
                    float lb=0;
                    int idx = i * spatial_dim + j;
                    lb = mask_data[idx];
                  
                     // -------------------- clear ---------------------//
                    if ( lb > 0) {
                       for (int c = 0; c < channels; ++c) {
                          diff_data[i * dim + c * spatial_dim + j] = 0;
                       }
                    }
                    // -------------------- clear ---------------------//
                }
              }
        }
        /*if(this->layer_param_.manipulate_loss_param().uniform_droprate()<1.0)
        {
            float rd = ((caffe::caffe_rng_rand() % INT_MAX)*1.0 / (INT_MAX-1));
            if ( rd > this->layer_param_.manipulate_loss_param().uniform_droprate()) {
               for (int c = 0; c < channels; ++c) {
                  diff_data[i * dim + c * spatial_dim + j] = 0;
               }
            }
        }*/
        /*
        if (propagate_down[0]) {                                                
            int propagate_count = 0;                                            
            for (int i = 0; i < num; ++i)                                       
                for (int j = 0; j < spatial_dim; ++j) {                         
                    if (std::abs(diff_data[i * dim + j]) > 1e-10)               
                        ++propagate_count;                                      
                }                                                               
            caffe_scal(diff_.count(), Dtype(dim * spatial_dim) / propagate_count, diff_.mutable_cpu_diff());
            caffe_copy(diff_.count(), diff_.cpu_diff(), bottom[0]->mutable_cpu_diff());                                              
        }*/
        
        if (propagate_down[0]) {
            caffe_copy(diff_.count(), diff_.cpu_diff(), bottom[0]->mutable_cpu_diff());
        }
}

#ifdef CPU_ONLY
STUB_GPU(ManipulateLossLayer);
#endif

INSTANTIATE_CLASS(ManipulateLossLayer);
REGISTER_LAYER_CLASS(ManipulateLoss);

}  // namespace caffe
