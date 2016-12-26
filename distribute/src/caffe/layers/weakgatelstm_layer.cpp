/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/weakgatelstm_layer.hpp"

namespace caffe {

template <typename Dtype>
void WeakGateLstmLayer<Dtype>::disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels)
{
 
    const int dims []= {num_,channels,height_,width_};
    const int horizontal_disorder []= {3,0,2,1};
    const int vertical_disorder []= {2,0,3,1};
    const int dimsize = 4;
    if(horizontal_ && ! reverse_)
    {// left --> right
        caffe_cpu_permute(datain,dataout,dims,horizontal_disorder,dimsize,-1);
    }
    else if(horizontal_ &&  reverse_)
    {// right --> left
        caffe_cpu_permute(datain,dataout,dims,horizontal_disorder,dimsize,0);
    }
    else if( !horizontal_ && !reverse_)
    {// top --> bottom
        caffe_cpu_permute(datain,dataout,dims,vertical_disorder,dimsize,-1);
    }
    else
    {// bottom --> top
        caffe_cpu_permute(datain,dataout,dims,vertical_disorder,dimsize,0);
    }
    return;
}
template <typename Dtype>
void WeakGateLstmLayer<Dtype>::reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels)
{
    
    
    const int horizontal_recoverdims []= {width_,num_,height_,channels};
    const int vertical_recoverdims []= {height_,num_,width_,channels};
    const int horizontal_recoverorder []= {1,3,2,0};
    const int vertical_recoverorder []= {1,3,0,2};
    const int dimsize = 4;
    
    if(horizontal_ && ! reverse_)
    {// left --> right
        caffe_cpu_permute(datain,dataout,horizontal_recoverdims,horizontal_recoverorder,dimsize,-1);
    }
    else if(horizontal_ &&  reverse_)
    {// right --> left
        caffe_cpu_permute(datain,dataout,horizontal_recoverdims,horizontal_recoverorder,dimsize,3);
    }
    else if( !horizontal_ && !reverse_)
    {// top --> bottom
        caffe_cpu_permute(datain,dataout,vertical_recoverdims,vertical_recoverorder,dimsize,-1);
    }
    else
    {// bottom --> top
        caffe_cpu_permute(datain,dataout,vertical_recoverdims,vertical_recoverorder,dimsize,2);
    }
    return;
}

template <typename Dtype>
void WeakGateLstmLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        
        channels_ = bottom[0]->channels();
        height_out_ = bottom[0]->height();
        width_out_ = bottom[0]->width();

        num_output_ = this->layer_param_.weakgatelstm_param().num_output();
        horizontal_ = this->layer_param_.weakgatelstm_param().horizontal();
        reverse_ = this->layer_param_.weakgatelstm_param().reverse();
        bias_term_ = this->layer_param_.weakgatelstm_param().bias_term();
        restrict_w_ = this->layer_param_.weakgatelstm_param().restrict_w();
        
        CHECK(bottom.size() ==2 )<<"bottom size can only be 2";   

        int gate_channels = bottom[1]->channels();
        //CHECK(gate_channels_ == 1 || gate_channels_ == num_output_)<<"gate channels can only be 1 or equal to top channels";
        CHECK(gate_channels == num_output_)<<"gate channels can only equal to top channels";
        
        if (this->blobs_.size() > 0) 
        {
            LOG(INFO) << "Skipping parameter initialization";
        } 
        else {
            if (bias_term_)
            {
                this->blobs_.resize(3);
            } 
            else
            {
                this->blobs_.resize(2);
            }

            //Wx 
            this->blobs_[0].reset(new Blob<Dtype>(1, 1, 2 * num_output_, channels_));
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                this->layer_param_.weakgatelstm_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());
            
            //Wh
            this->blobs_[1].reset(new Blob<Dtype>(1, 1, 2 * num_output_, num_output_));
            weight_filler->Fill(this->blobs_[1].get());

            if (bias_term_)
            {
                //b  
                this->blobs_[2].reset(new Blob<Dtype>(1, 1, 2 * num_output_,1));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                    this->layer_param_.weakgatelstm_param().bias_filler()));
                bias_filler->Fill(this->blobs_[2].get());
            }
        }

        this->param_propagate_down_.resize(this->blobs_.size(), true);
        
        identical_multiplier_.Reshape(1, 1, num_output_,num_output_);
        Dtype* id_data = identical_multiplier_.mutable_cpu_data();
        for (int h=0;h<num_output_;h++)
        {
            for(int w=0;w<num_output_;w++)
            {
                if(w == h)
                {
                    id_data[h*num_output_ + w] = Dtype(1);
                }
                else
                {
                    id_data[h*num_output_ + w] = Dtype(0);
                }
            }
        }
    //print FLPS
    Dtype flps = 0.0; 
    num_ = bottom[0]->num();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    if(this->horizontal_)
    {
        T_ = width_;
        col_length_ = height_;
    }
    else
    {
        T_ = height_;
        col_length_ = width_; 
    }
    //w*x
    flps += 2* num_output_ * channels_ *(T_ -1) * col_length_ *2;
    //w*h
    flps += 2* num_output_ * num_output_ *(T_ -1) * col_length_ *2;
    //+b
    flps += num_output_ * col_length_ *(T_ -1)*2;
    //add wx and (wh+b)
    flps += num_output_ * col_length_ *(T_ -1)*2;
    //c=i.*u+ g.*c
    flps += num_output_ * col_length_ *(T_ -1)*2;
    //h=c
 
    flps = flps/1000/1000/1000;
    LOG(INFO)<<this->layer_param_.name()<<" type: "<<this->layer_param_.type()<<" FPLS(G):"<<flps;
}

template <typename Dtype>
void WeakGateLstmLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
        num_ = bottom[0]->num();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        CHECK_EQ(bottom[0]->channels(), channels_) << "Input channels incompatible ";
        CHECK_EQ(height_, height_out_) << "Input height incompatible ";
        CHECK_EQ(width_, width_out_) << "Input width incompatible ";

        if(this->horizontal_)
        {
            T_ = width_;
            col_length_ = height_;
        }
        else
        {
            T_ = height_;
            col_length_ = width_; 
        }
        x_col_count_ = col_length_ * num_ * channels_;
        h_col_count_ = col_length_ * num_ * num_output_;
    
        // TODO: generalize to handle inputs of different shapes.
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) 
        {
            CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
            
            CHECK_EQ(height_, bottom[bottom_id]->height())
            << "Inputs must have same height.";
            CHECK_EQ(width_, bottom[bottom_id]->width())
            << "Inputs must have same width.";
        }

        // Shape the tops.
        for (int top_id = 0; top_id < top.size(); ++top_id) 
        {
            top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
        }

        
        L_buffer_.Reshape(2 * num_ , num_output_ , height_ , width_);
        C_buffer_.Reshape(num_ , num_output_ , height_ , width_);
        G_buffer_.Reshape(num_ , num_output_ , height_ , width_);
        
        H_buffer_.Reshape(num_ , num_output_ , height_, width_);
        X_buffer_.Reshape(num_ , channels_ , height_, width_);
        
        GL_buffer_.Reshape(1 , 1 ,  num_output_ , num_ * col_length_);
        Trans_buffer_.Reshape(1 , 1 , num_output_ , num_ * col_length_);
        Ct_active_buffer_.Reshape(1 , 1 , num_output_ , num_ * col_length_);
        
        if (bias_term_) 
        {
            bias_multiplier_.Reshape(1, 1, 1, num_ * col_length_);
            caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
}

template <typename Dtype>
void WeakGateLstmLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
    if(restrict_w_ > 0)
    {
        caffe_bound(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), Dtype(-restrict_w_), Dtype(restrict_w_), this->blobs_[0]->mutable_cpu_data());
        caffe_bound(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), Dtype(-restrict_w_), Dtype(restrict_w_), this->blobs_[1]->mutable_cpu_data());
    }
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_gate = bottom[1]->cpu_data();
    const Dtype* W_x = this->blobs_[0]->cpu_data();
    const Dtype* W_h = this->blobs_[1]->cpu_data();
    const Dtype* bias_data = this->blobs_[2]->cpu_data();
       
    Dtype* X_data = X_buffer_.mutable_cpu_data();
    Dtype* H_data = H_buffer_.mutable_cpu_data();
    Dtype* G_data = G_buffer_.mutable_cpu_data();
    Dtype* L_data = L_buffer_.mutable_cpu_data();
    Dtype* C_data = C_buffer_.mutable_cpu_data();
    Dtype* GL_mid_data = GL_buffer_.mutable_cpu_data();
    Dtype* Trans_data = Trans_buffer_.mutable_cpu_data();
    Dtype* ct_active = Ct_active_buffer_.mutable_cpu_data();
       
    //get X_data
    disorder_inputdata((const Dtype *)bottom_data,X_data,horizontal_,reverse_,channels_);

    //get gate data
    disorder_inputdata((const Dtype *)bottom_gate,G_data,horizontal_,reverse_,num_output_);
    
    M_ = 2 * num_output_;
    N_ = num_ * col_length_;
    K_x_ = channels_;
    K_h_ = num_output_;
    
    int L_count = M_ * N_;
    int G_count = h_col_count_;
    int C_count = h_col_count_;
    //int bias_count = M_;
    
    for(int t=0; t < T_; t++)
    {//finish left to right gate lstm in this loop
    
        Dtype* L_t = L_data + t * L_count;
        Dtype* G_t = G_data + t * G_count;
        Dtype* C_t = C_data + t * C_count;
        Dtype* i_t = L_t;
        Dtype* u_t = L_t + h_col_count_;
       

        //L(t) = b
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1,
                (Dtype)1.,bias_data, bias_multiplier_.cpu_data(), 
                (Dtype)0., L_t );
                
        //L(t) += W_x * X(t)' 
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_x_,
                (Dtype)1.,W_x, X_data + t * x_col_count_, 
                (Dtype)1., L_t );
           
        //L(t) += W_h * H(t-1)' if t > 0
        if(t > 0)
        {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_h_,
                (Dtype)1.,W_h, H_data+ (t-1)* h_col_count_, 
                (Dtype)1., L_t );
        }
        
        //active L(t) --> i u
        caffe_cpu_sigmoid_forward(h_col_count_,i_t,i_t);
        caffe_cpu_tanh_forward(h_col_count_,u_t,u_t);
        
        //transpose G(t) to  trans_data , now trans_data is G(t)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1.,identical_multiplier_.cpu_data(), G_t, 
                    (Dtype)0., Trans_data);
                    
        //C(t) = i(t) .* u(t)
        caffe_mul<Dtype>(h_col_count_, i_t, u_t, C_t);
        
        //C(t) += G(t)' .* C(t-1)  if t >0 
        if(t>0)
        {
            //temp save G(t)' .* C(t-1)  in GL_mid_data
            caffe_mul<Dtype>(C_count, Trans_data, C_data + (t-1)*C_count, GL_mid_data);
            //then add to C(t)
            caffe_add<Dtype>(C_count, GL_mid_data, C_t , C_t );
        }
        
        //active C(t)
        caffe_cpu_tanh_forward(C_count, C_t, ct_active);
        
        //transpose tand[C(t)] to H(t)
        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_h_, K_h_,
                    (Dtype)1.,ct_active, identical_multiplier_.cpu_data(),
                    (Dtype)0., H_data + t * h_col_count_);
    }
    //then recover order to top data
    reorder_outputdata((const Dtype *)H_data,top[0]->mutable_cpu_data(),horizontal_,reverse_,num_output_);
}

template <typename Dtype>
void WeakGateLstmLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    M_ = 2 * num_output_;
    N_ = num_ * col_length_;
    K_x_ = channels_;
    K_h_ = num_output_;
    
    int L_count = M_ * N_;
    int G_count = h_col_count_;
    int C_count = h_col_count_;
     
    const Dtype* W_x = NULL;
    Dtype* W_x_diff = NULL;
    const Dtype* W_h = NULL;
    Dtype* W_h_diff = NULL;

    //clear w diff and b diff
    if (this->param_propagate_down_[0]) {
        W_x = this->blobs_[0]->cpu_data();
        W_x_diff = this->blobs_[0]->mutable_cpu_diff();
        //caffe_set(this->blobs_[0]->count(), Dtype(0), W_x_diff);
    }
    if (this->param_propagate_down_[1]) {
        W_h = this->blobs_[1]->cpu_data();
        W_h_diff = this->blobs_[1]->mutable_cpu_diff();
        //caffe_set(this->blobs_[1]->count(), Dtype(0), W_h_diff);
    }
    Dtype* bias_diff = NULL;
    if (bias_term_ && this->param_propagate_down_[2]) {
        bias_diff = this->blobs_[2]->mutable_cpu_diff();
        //caffe_set(this->blobs_[2]->count(), Dtype(0), bias_diff);
        
    } 
    
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* C_data = C_buffer_.cpu_data();
    const Dtype* L_data = L_buffer_.cpu_data();
    const Dtype* G_data = G_buffer_.cpu_data();
    const Dtype* X_data = X_buffer_.cpu_data();
    const Dtype* H_data = H_buffer_.cpu_data();
    
    Dtype* H_diff = H_buffer_.mutable_cpu_diff();
    Dtype* X_diff = X_buffer_.mutable_cpu_diff();
    Dtype* C_diff = C_buffer_.mutable_cpu_diff();
    Dtype* L_diff = L_buffer_.mutable_cpu_diff();
    Dtype* G_diff = G_buffer_.mutable_cpu_diff();
    
    Dtype* Trans_diff = Trans_buffer_.mutable_cpu_diff();
    Dtype* Trans_G = Trans_buffer_.mutable_cpu_data();
    Dtype* ct_active = Ct_active_buffer_.mutable_cpu_data();
    //Dtype* temp_PL_diff = GL_buffer_.mutable_cpu_diff();
  
    

    //H(i)_diff = top_diff
    disorder_inputdata((const Dtype *)top_diff,H_diff,horizontal_,reverse_,num_output_);

    for(int t= T_ - 1; t >= 0; t--)
    {//finish right to left gate lstm BP in this loop
        const Dtype* G_t = G_data + t*G_count;
        const Dtype* C_t = C_data + t*C_count;
        const Dtype* L_t = L_data + t*L_count;
        const Dtype* i_t = L_t ;
        const Dtype* u_t = L_t + h_col_count_;
        
        Dtype* L_t_diff = L_diff + t*L_count;
        Dtype* G_t_diff = G_diff + t*G_count;
        Dtype* C_t_diff = C_diff + t*C_count;
        Dtype* i_t_diff = L_t_diff;
        Dtype* u_t_diff = L_t_diff + h_col_count_;

        //H(t)_diff += L(t+1)_diff' * W_h  if t < T-1
        if(t < T_-1)
        {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_h_, M_,
                    (Dtype)1., L_diff + (t+1)*L_count,  W_h,
                    (Dtype)1., H_diff + t * h_col_count_);
        }
        
        //transpose H(t)_diff, now Trans_diff is H(t)_diff'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1., identical_multiplier_.cpu_data(), H_diff + t * h_col_count_ ,
                    (Dtype)0., Trans_diff);
        
        //C(t)_diff = H(t)_diff'  ==>> activeback[C(t)]
        caffe_cpu_tanh_forward(C_count, C_t, ct_active);
        caffe_cpu_tanh_backward(h_col_count_,ct_active,Trans_diff,C_t_diff);
        
        //transpose G(t+1), now Trans_G is G(t+1)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1., identical_multiplier_.cpu_data(), G_data + (t+1) * G_count ,
                    (Dtype)0., Trans_G);
                    
        //C(t)_diff  += C(t+1)_diff .* G(t+1)'  if t < T-1
        if(t < T_-1)
        {
            //save C(t+1)_diff .* G(t+1)' in ct_active
            caffe_mul<Dtype>(C_count, C_diff + (t+1)*C_count, Trans_G, ct_active);
            // then add to C(t)_diff
            caffe_add<Dtype>(C_count, ct_active, C_t_diff, C_t_diff);
        }
        
        //i(t)_diff = C(t)_diff .* u(t)
        caffe_mul<Dtype>(h_col_count_, C_t_diff, u_t, i_t_diff);
        
        //u(t)_diff = C(t)_diff .* i(t)
        caffe_mul<Dtype>(h_col_count_, C_t_diff, i_t, u_t_diff);
        
        //active back L(t)_diff  --> i, u
        caffe_cpu_sigmoid_backward(h_col_count_, i_t, i_t_diff,i_t_diff);
        caffe_cpu_tanh_backward(h_col_count_, u_t, u_t_diff,u_t_diff);
        
        //transpose G(t), now Trans_G is G(t)'
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_h_, N_, K_h_,
                    (Dtype)1., identical_multiplier_.cpu_data(), G_t ,
                    (Dtype)0., Trans_G);
                    
       
        
        if(propagate_down[1])
        {
            //G(t)_diff' = C(t)_diff .*C(t-1) if t > 0
            if(t>0)
            {
                //save C(t)_diff .*C(t-1) in Trans_G, now Trans_G is G(t)_diff' 
                caffe_mul<Dtype>(h_col_count_, C_t_diff, C_data + (t-1)*C_count, Trans_G);

                // transpose G(t)_diff' to G(t)_diff
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_h_, K_h_,
                            (Dtype)1.,Trans_G, identical_multiplier_.cpu_data(),
                            (Dtype)0., G_t_diff);
            }
        }     
        if (this->param_propagate_down_[0])
        {        
            //W_x_diff += L(t)_diff * X(t)
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_x_, N_,
                        (Dtype)1., L_t_diff, X_data + t * x_col_count_,
                        (Dtype)1., W_x_diff);
        }
        if (this->param_propagate_down_[1])
        {
            //W_h_diff += L(t)_diff * H(t-1) if t >0
            if(t>0)
            {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_h_, N_,
                            (Dtype)1., L_t_diff, H_data + (t-1) * h_col_count_,
                            (Dtype)1., W_h_diff);
            }
        }
        if(propagate_down[0])
        {
            //X(t)_diff = L(t)_diff' * W_x
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_x_, M_,
                        (Dtype)1., L_t_diff, W_x,
                        (Dtype)0., X_diff + t * x_col_count_);     
        }
        if (this->param_propagate_down_[2])
        {
            //b(t)_diff += L(t)_diff
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 1, N_,
                               (Dtype)1.,L_t_diff, bias_multiplier_.cpu_data(), 
                               (Dtype)1., bias_diff);
        }
    }
    if(propagate_down[0])
    {
        reorder_outputdata((const Dtype *)X_diff,bottom[0]->mutable_cpu_diff(),horizontal_,reverse_,channels_);
    }
    if(propagate_down[1])
    {
        reorder_outputdata((const Dtype *)G_diff,bottom[1]->mutable_cpu_diff(),horizontal_,reverse_,num_output_);
    }
    
    if(this->layer_param_.weakgatelstm_param().printall_datadiff())
    {
        //print data
        bottom[0]->run_statistics();
        LOG(INFO) << this->layer_param().name()<< " bottom[0] data:"
                  << " min=" << bottom[0]->min_data() << " max=" << bottom[0]->max_data()<<" mean="<<bottom[0]->mean_data()
                  <<" asum="<<bottom[0]->asum_data()<<" std="<<bottom[0]->std_data();
        LOG(INFO) << this->layer_param().name()<< " bottom[0] diff:"
                  << " min=" << bottom[0]->min_diff() << " max=" << bottom[0]->max_diff()<<" mean="<<bottom[0]->mean_diff()
                  <<" asum="<<bottom[0]->asum_diff()<<" std="<<bottom[0]->std_diff();
        
        //print gate
        bottom[1]->run_statistics();
        LOG(INFO) << this->layer_param().name()<< " bottom[1] data:"
                  << " min=" << bottom[1]->min_data() << " max=" << bottom[1]->max_data()<<" mean="<<bottom[1]->mean_data()
                  <<" asum="<<bottom[1]->asum_data()<<" std="<<bottom[1]->std_data();
        LOG(INFO) << this->layer_param().name()<< " bottom[1] diff:"
                  << " min=" << bottom[1]->min_diff() << " max=" << bottom[1]->max_diff()<<" mean="<<bottom[1]->mean_diff()
                  <<" asum="<<bottom[1]->asum_diff()<<" std="<<bottom[1]->std_diff();
        
        //print L
        L_buffer_.run_statistics();
        LOG(INFO) << this->layer_param().name()<< " L data:"
                  << " min=" << L_buffer_.min_data() << " max=" << L_buffer_.max_data()<<" mean="<<L_buffer_.mean_data()
                  <<" asum="<<L_buffer_.asum_data()<<" std="<<L_buffer_.std_data();
        LOG(INFO) << this->layer_param().name()<< " L diff:"
                  << " min=" << L_buffer_.min_diff() << " max=" << L_buffer_.max_diff()<<" mean="<<L_buffer_.mean_diff()
                  <<" asum="<<L_buffer_.asum_diff()<<" std="<<L_buffer_.std_diff();
                  
        /*//print P
        P_buffer_.run_statistics();
        LOG(INFO) << this->layer_param().name()<< " P data:"
                  << " min=" << P_buffer_.min_data() << " max=" << P_buffer_.max_data()<<" mean="<<P_buffer_.mean_data()
                  <<" asum="<<P_buffer_.asum_data()<<" std="<<P_buffer_.std_data();
        LOG(INFO) << this->layer_param().name()<< " P diff:"
                  << " min=" << P_buffer_.min_diff() << " max=" << P_buffer_.max_diff()<<" mean="<<P_buffer_.mean_diff()
                  <<" asum="<<P_buffer_.asum_diff()<<" std="<<P_buffer_.std_diff();
        */          
        //print C
        C_buffer_.run_statistics();
        LOG(INFO) << this->layer_param().name()<< " C data:"
                  << " min=" << C_buffer_.min_data() << " max=" << C_buffer_.max_data()<<" mean="<<C_buffer_.mean_data()
                  <<" asum="<<C_buffer_.asum_data()<<" std="<<C_buffer_.std_data();
        LOG(INFO) << this->layer_param().name()<< " C diff:"
                  << " min=" << C_buffer_.min_diff() << " max=" << C_buffer_.max_diff()<<" mean="<<C_buffer_.mean_diff()
                  <<" asum="<<C_buffer_.asum_diff()<<" std="<<C_buffer_.std_diff();
               
        //print G
        G_buffer_.run_statistics();
        LOG(INFO) << this->layer_param().name()<< " G data:"
                  << " min=" << G_buffer_.min_data() << " max=" << G_buffer_.max_data()<<" mean="<<G_buffer_.mean_data()
                  <<" asum="<<G_buffer_.asum_data()<<" std="<<G_buffer_.std_data();
        LOG(INFO) << this->layer_param().name()<< " G diff:"
                  << " min=" << G_buffer_.min_diff() << " max=" << G_buffer_.max_diff()<<" mean="<<G_buffer_.mean_diff()
                  <<" asum="<<G_buffer_.asum_diff()<<" std="<<G_buffer_.std_diff();
                  
        //print param[0]
        this->blobs_[0]->run_statistics();
        LOG(INFO) << this->layer_param().name()<< " param[0] data:"
                  << " min=" << this->blobs_[0]->min_data() << " max=" << this->blobs_[0]->max_data()<<" mean="<<this->blobs_[0]->mean_data()
                  <<" asum="<<this->blobs_[0]->asum_data()<<" std="<<this->blobs_[0]->std_data();
        LOG(INFO) << this->layer_param().name()<< " param[0] diff:"
                  << " min=" << this->blobs_[0]->min_diff() << " max=" << this->blobs_[0]->max_diff()<<" mean="<<this->blobs_[0]->mean_diff()
                  <<" asum="<<this->blobs_[0]->asum_diff()<<" std="<<this->blobs_[0]->std_diff();
        
        //print param[1]
        this->blobs_[1]->run_statistics();
        LOG(INFO) << this->layer_param().name()<< " param[0] data:"
                  << " min=" << this->blobs_[1]->min_data() << " max=" << this->blobs_[1]->max_data()<<" mean="<<this->blobs_[1]->mean_data()
                  <<" asum="<<this->blobs_[1]->asum_data()<<" std="<<this->blobs_[1]->std_data();
        LOG(INFO) << this->layer_param().name()<< " param[0] diff:"
                  << " min=" << this->blobs_[1]->min_diff() << " max=" << this->blobs_[1]->max_diff()<<" mean="<<this->blobs_[1]->mean_diff()
                  <<" asum="<<this->blobs_[1]->asum_diff()<<" std="<<this->blobs_[1]->std_diff();
                  
        //print param[2]
        this->blobs_[2]->run_statistics();
        LOG(INFO) << this->layer_param().name()<< " param[0] data:"
                  << " min=" << this->blobs_[2]->min_data() << " max=" << this->blobs_[2]->max_data()<<" mean="<<this->blobs_[2]->mean_data()
                  <<" asum="<<this->blobs_[2]->asum_data()<<" std="<<this->blobs_[2]->std_data();
        LOG(INFO) << this->layer_param().name()<< " param[0] diff:"
                  << " min=" << this->blobs_[2]->min_diff() << " max=" << this->blobs_[2]->max_diff()<<" mean="<<this->blobs_[2]->mean_diff()
                  <<" asum="<<this->blobs_[2]->asum_diff()<<" std="<<this->blobs_[2]->std_diff();
       
    }
}

#ifdef CPU_ONLY
STUB_GPU(WeakGateLstmLayer);
#endif

INSTANTIATE_CLASS(WeakGateLstmLayer);
REGISTER_LAYER_CLASS(WeakGateLstm);

}  // namespace caffe
