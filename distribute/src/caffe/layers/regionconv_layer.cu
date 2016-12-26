/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/regionconv_layer.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void RegionconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    /*
    bool debug=false;
    FILE * fp;
   std::string filename;
   if(debug)
   {
    //temp debug
   caffe_gpu_set(bottom[0]->count(), Dtype(0.1), bottom[0]->mutable_gpu_data());
   caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.1), this->blobs_[0]->mutable_gpu_data());
   caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.1), this->blobs_[1]->mutable_gpu_data());
   
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_bottomdata.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = bottom[0]->cpu_data();
       int count = bottom[0]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_w.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = this->blobs_[0]->cpu_data();
       int count = this->blobs_[0]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_b.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = this->blobs_[1]->cpu_data();
       int count = this->blobs_[1]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   }
   */
    for (int i = 0; i < bottom.size(); ++i) {
        
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        Dtype* input_col_data = input_col_buffer_.mutable_gpu_data();
        Dtype* output_col_data = output_col_buffer_.mutable_gpu_data();
        const Dtype* weight = this->blobs_[0]->gpu_data();
        
        for (int n = 0; n < num_; ++n) {
            if(input_is_1x1_)
            {
                input_col_data = (Dtype*)(bottom_data + bottom[i]->offset(n));
            }
            else
            {
                im2col_gpu(bottom_data + bottom[i]->offset(n), input_channels_, input_height_,
                          input_width_, input_patch_h_, input_patch_w_, input_pad_h_, input_pad_w_, input_stride_h_, input_stride_w_,
                          input_col_data);    
            }
            
            if(output_is_1x1_)
            {
                output_col_data = top_data + top[i]->offset(n);
            }
            for (int g = 0; g < group_; ++g) {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
                  (Dtype)1., weight + weight_offset_ * g , input_col_data + input_col_offset_ * g,
                  (Dtype)0., output_col_data + output_col_offset_ * g);
            }
            
            if(!output_is_1x1_)
            {
                col2im_gpu(output_col_data, num_output_, height_out_, width_out_,
                        output_patch_h_, output_patch_w_, output_pad_h_, output_pad_w_,
                        output_stride_h_, output_stride_w_, top_data + top[i]->offset(n));
            }
   
            if (bias_term_) {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                    height_out_ * width_out_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
                    bias_multiplier_.gpu_data(),
                    (Dtype)1., top_data + top[i]->offset(n));
            }  
        }
        
      }
      /*
    if(debug)
    {
     //tempdebug
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_topdata.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = top[0]->cpu_data();
       int count = top[0]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   }*/
}

template <typename Dtype>
void RegionconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /*
    bool debug=false;
    FILE * fp;
   std::string filename;
   if(debug)
   {
    //temp debug
   caffe_gpu_set(top[0]->count(), Dtype(0.1), top[0]->mutable_gpu_diff());
   caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.1), this->blobs_[0]->mutable_gpu_data());
   caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.1), this->blobs_[1]->mutable_gpu_data());
   
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_topdiff.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = top[0]->cpu_data();
       int count = top[0]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   }
   */
    const Dtype* weight = NULL;
      Dtype* weight_diff = NULL;
      if (this->param_propagate_down_[0]) {
        weight = this->blobs_[0]->gpu_data();
        weight_diff = this->blobs_[0]->mutable_gpu_diff();
        //caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
      }
      Dtype* bias_diff = NULL;
      if (bias_term_ && this->param_propagate_down_[1]) {
        bias_diff = this->blobs_[1]->mutable_gpu_diff();
        //caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
      }

      for (int i = 0; i < top.size(); ++i) {
      
        const Dtype* top_diff = NULL;
        // Bias gradient, if necessary.
        if (bias_term_ && this->param_propagate_down_[1]) {
          top_diff = top[i]->gpu_diff();
          for (int n = 0; n < num_; ++n) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                     1,height_out_ * width_out_, (Dtype)1.,top_diff + top[i]->offset(n) ,
                    bias_multiplier_.gpu_data(),
                    (Dtype)1., bias_diff);
          }
        }
        if (this->param_propagate_down_[0] || propagate_down[i]) {
          if (!top_diff) {
            top_diff = top[i]->gpu_diff();
          }
          
          Dtype* input_col_data = input_col_buffer_.mutable_gpu_data();
          Dtype* input_col_diff = input_col_buffer_.mutable_gpu_diff();
          Dtype* output_col_data = output_col_buffer_.mutable_gpu_data();
          Dtype* output_col_diff = output_col_buffer_.mutable_gpu_diff();

          const Dtype* bottom_data = bottom[i]->gpu_data();
          Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
          
          
          for (int n = 0; n < num_; ++n) {
            
            if(input_is_1x1_)
            {
                input_col_data = (Dtype*)(bottom_data + bottom[i]->offset(n));
            }
            else
            {
                im2col_gpu(bottom_data + bottom[i]->offset(n), input_channels_, input_height_,
                          input_width_, input_patch_h_, input_patch_w_, input_pad_h_, input_pad_w_, input_stride_h_, input_stride_w_,
                          input_col_data);   
            }   

            if(output_is_1x1_)
            {
                output_col_diff = (Dtype*)(top_diff + top[i]->offset(n));
            }
            else
            {
                im2col_gpu(top_diff + top[i]->offset(n), num_output_, height_out_,
                           width_out_, output_patch_h_, output_patch_w_, output_pad_h_, output_pad_w_, output_stride_h_, output_stride_w_,
                          output_col_diff);  
            }                         
            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              for (int g = 0; g < group_; ++g) {
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                    (Dtype)1., output_col_diff + output_col_offset_ * g,
                    input_col_data + input_col_offset_ * g, (Dtype)1.,
                    weight_diff + weight_offset_ * g);
              }
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
                if (weight == NULL) {
                weight = this->blobs_[0]->gpu_data();
                }
                if(input_is_1x1_)
                {
                    input_col_diff = (Dtype *)(bottom_diff + bottom[i]->offset(n));
                }
                for (int g = 0; g < group_; ++g) {
                    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                        (Dtype)1., weight + weight_offset_ * g,
                        output_col_diff + output_col_offset_ * g,
                        (Dtype)0., input_col_diff + input_col_offset_ * g);
                }
                if(!input_is_1x1_)
                {
                    // col2im back to the data                
                    col2im_gpu(input_col_diff, input_channels_, input_height_, input_width_,
                        input_patch_h_, input_patch_w_, input_pad_h_, input_pad_w_,
                        input_stride_h_, input_stride_w_, bottom_diff + bottom[i]->offset(n));
                }
            }
          }
        }
      }
    /*
    if(debug)
    {
    fp = NULL;
   filename =  this->layer_param().name();
   filename += "_bottomdiff.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = bottom[0]->cpu_diff();
       int count = bottom[0]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_w_diff.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = this->blobs_[0]->cpu_diff();
       int count = this->blobs_[0]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   
   fp = NULL;
   filename =  this->layer_param().name();
   filename += "_b_diff.txt";
   fp=fopen(filename.c_str(),"r");
   if(fp == NULL)
   {
       fp=fopen(filename.c_str(),"w");
       const Dtype* inputdata = this->blobs_[1]->cpu_diff();
       int count = this->blobs_[1]->count();
       for(int tempi=0;tempi<count;tempi++)
       {
           fprintf(fp,"%f\n",(float)inputdata[tempi]);
       }
       fclose(fp);
   }
   else
   {
    fclose(fp);
   }
   }
   */
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionconvolutionLayer);

}  // namespace caffe
