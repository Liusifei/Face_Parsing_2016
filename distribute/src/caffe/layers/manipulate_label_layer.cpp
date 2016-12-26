/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/manipulatelabel_layer.hpp"

namespace caffe {

template <typename Dtype>
void ManipulateLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
    horizontal_ = this->layer_param_.manipulate_label_param().horizontal();
    edgerange_ = this->layer_param_.manipulate_label_param().edgerange();
    maxlabel_ = this->layer_param_.manipulate_label_param().maxlabel();
    duplicate_dim_ = this->layer_param_.manipulate_label_param().duplicate_dim();
    duplicate_num_ = this->layer_param_.manipulate_label_param().duplicate_num();

}

template <typename Dtype>
void ManipulateLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        
        switch (this->layer_param_.manipulate_label_param().type()) 
        {
            case ManipulatelabelParameter_Type_ANYONE:
            case ManipulatelabelParameter_Type_EDGE:
                top[0]->Reshape(num, channels,height, width);
                break;
            case ManipulatelabelParameter_Type_EXPAND:
                
                CHECK(channels == 1) << "input of ManipulateLabelLayer (type:expand) must have only 1 channels, but here get "<<channels;
                top[0]->Reshape(num, maxlabel_+1,height, width);
                break;
            case ManipulatelabelParameter_Type_DUPLICATE:
                //CHECK(false)<<"not implment yet!";
                CHECK(duplicate_num_ > 1);
                CHECK(duplicate_dim_==0 || duplicate_dim_==1)<<"duplicate dim can only be 0 or 1, here get "<<duplicate_dim_;
                if(duplicate_dim_==0)
                {
                    top[0]->Reshape(num * duplicate_num_, channels,height, width);
                }
                if(duplicate_dim_==1)
                {
                    top[0]->Reshape(num, channels * duplicate_num_,height, width);
                }
                break;
            default:
                LOG(FATAL) << "Unknown type method.";
        }
}

template <typename Dtype>
void ManipulateLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int height = bottom[0]->height();
        int width = bottom[0]->width();
        const int count = top[0]->count();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        
        const Dtype* label = bottom[0]->cpu_data();
        int spatial_dim = height * width;        
        int top_dim = top[0]->count() / top[0]->num(); 
        int bottom_dim = bottom[0]->count() / bottom[0]->num(); 
        int dim = top_dim;
        bool reverselabel = this->layer_param_.manipulate_label_param().reverse_label();
        Dtype edgevalue = 1;
        caffe_set(count, Dtype(0), top_data);
     
        switch (this->layer_param_.manipulate_label_param().type()) 
        {
            case ManipulatelabelParameter_Type_ANYONE:
               // LOG(INFO)<<"bottom[0]->asum_data()>= "<<bottom[0]->asum_data();
                for(int n=0;n< bottom[0]->num();n++)
                {
                    if(caffe_cpu_asum(bottom_dim,bottom_data + n*bottom_dim)>=1)
                    {
                        caffe_set(top_dim,Dtype(1),top_data + n * top_dim);
                    }
                }
                break;
            case ManipulatelabelParameter_Type_EDGE:
                CHECK(edgerange_>0)<<"edge range must bigger than 0.";
                
                if(reverselabel)
                {
                    caffe_set(count, Dtype(1), top_data);
                    edgevalue = 0;
                }
                int new_h,new_w;
                for(int n=0;n<num;n++)
                    for(int c=0;c<channels;c++)
                        for(int h=0;h<height;h++)
                            for(int w=0;w<width;w++)
                            {
                                Dtype v = label[n*dim + c*spatial_dim + h*width + w];
                                bool isedge=false;
                                if(this->layer_param_.manipulate_label_param().both_edge_direction())
                                {
                                    for(int row = -1*edgerange_ ; row <= edgerange_;row++)
                                    {
                                        for(int col = -1*edgerange_ ; col <= edgerange_;col++)
                                        {
                                            new_h = h + row;
                                            new_w = w + col;
                                            if(new_h>=0 && new_h <height && new_w>=0 && new_w < width)
                                            {
                                                if(label[n*dim + c*spatial_dim + new_h*width + new_w] != v)
                                                {
                                                    isedge = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if(isedge)
                                            break;  
                                    }
                                    
                                }
                                else
                                {
                                    for(int r = -1*edgerange_ ; r <= edgerange_;r++)
                                    {
                                        if(horizontal_)
                                        {
                                            new_h = h;
                                            new_w = w + r;
                                        }
                                        else
                                        {
                                            new_h = h + r;
                                            new_w = w;
                                        }
                                        if(new_h>=0 && new_h <height && new_w>=0 && new_w < width)
                                        {
                                            if(label[n*dim + c*spatial_dim + new_h*width + new_w] != v)
                                            {
                                                isedge = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if(isedge)
                                    top_data[n*dim + c*spatial_dim + h*width + w] = edgevalue;
                            }

                break;
            case ManipulatelabelParameter_Type_EXPAND:
                CHECK(dim == spatial_dim * (maxlabel_ +1))<<"ManipulateLabelLayer dim == spatial_dim * (maxlabel_ +1)";
                for (int i = 0; i < num; ++i) {
                    for (int j = 0; j < spatial_dim; j++) {            
                        int lb = static_cast<int>(label[i * spatial_dim + j]);
                        if(lb > maxlabel_)
                        {
                            CHECK(false) << "max label was set to be "<<maxlabel_<<", but here get "<<lb;
                        }                      
                        top_data[i * dim + lb * spatial_dim + j] = Dtype(1);
                    }
                }
                break;
            case ManipulatelabelParameter_Type_DUPLICATE:
                //CHECK(false)<<"not implment yet!";
                
                if(duplicate_dim_ == 0)
                {
                    for(int i=0; i<duplicate_num_; i++)
                    {
                        caffe_copy(bottom[0]->count(),bottom[0]->cpu_data(),top_data + i*bottom[0]->count());
                    }
                }
                if(duplicate_dim_ == 1)
                {
                    for(int n=0;n< bottom[0]->num();n++)
                    {
                        for(int c = 0; c < duplicate_num_ ; c++)
                        {
                            caffe_copy(bottom_dim,bottom_data + n * bottom_dim,top_data + n*top_dim + c *bottom_dim);
                        }
                    }
                }
                if(this->layer_param_.manipulate_label_param().duplicate_isanyone())
                {   
                    
                    for(int n=0;n< bottom[0]->num();n++)
                    {
                        if(caffe_cpu_asum(bottom_dim,bottom_data + n*bottom_dim)>=1)
                        {
                            caffe_set(top_dim,Dtype(1),top_data + n * top_dim);
                        }
                    }
                }
                break;
                
            default:
                LOG(FATAL) << "Unknown type method.";
        } 
}

template <typename Dtype>
void ManipulateLabelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        int top_dim = top[0]->count() / top[0]->num(); 
        int bottom_dim = bottom[0]->count() / bottom[0]->num(); 
                    
        switch (this->layer_param_.manipulate_label_param().type()) 
        {
            case ManipulatelabelParameter_Type_ANYONE:
                caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
                break;
            case ManipulatelabelParameter_Type_EDGE:
                caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
                break;
            case ManipulatelabelParameter_Type_EXPAND:
                caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
                break;
            case ManipulatelabelParameter_Type_DUPLICATE:
                    if(this->layer_param_.manipulate_label_param().duplicate_isanyone())
                    {
                        caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
                    }
                    else if(duplicate_dim_ == 0)
                    {
                        for(int i=0; i<duplicate_num_; i++)
                        {
                            caffe_add(bottom[0]->count(),top_diff + i *bottom[0]->count(), bottom_diff , bottom_diff);
                        }
                    }
                    else if(duplicate_dim_ == 1)
                    {
                        for(int n=0;n< bottom[0]->num();n++)
                        {
                            for(int c = 0; c < duplicate_num_ ; c++)
                            {
                                caffe_add(bottom_dim,top_diff + n*top_dim + c *bottom_dim, bottom_diff + n * bottom_dim,bottom_diff + n * bottom_dim);
                            }
                        }
                    }
                    else
                        CHECK(false);
                    
                break;
            default:
                    LOG(FATAL) << "Unknown type method.";
        }    
    }   
        
}

#ifdef CPU_ONLY
STUB_GPU(ManipulateLabelLayer);
#endif

INSTANTIATE_CLASS(ManipulateLabelLayer);
REGISTER_LAYER_CLASS(ManipulateLabel);

}  // namespace caffe
