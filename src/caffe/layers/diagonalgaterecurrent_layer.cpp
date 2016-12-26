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
#include "caffe/util/io.hpp"
#include "caffe/layers/diagonalgaterecurrent_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::active_Forward_cpu( const int n, Dtype * data )
	{
		switch ( this->layer_param_.gaterecurrent_param().active() )
		{
		case GateRecurrentParameter_Active_LINEAR:
			/* do nothing */
			break;
		case GateRecurrentParameter_Active_SIGMOID:
			caffe_cpu_sigmoid_forward( n, data, data );
			break;
		case GateRecurrentParameter_Active_RELU:
			caffe_cpu_relu_forward( n, data, data );
			break;
		case GateRecurrentParameter_Active_TANH:
			caffe_cpu_tanh_forward( n, data, data );
			break;
		default:
			LOG( FATAL ) << "Unknown active method.";
		}
	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::active_Backward_cpu( const int n, const Dtype * data, Dtype * diff )
	{
		switch ( this->layer_param_.gaterecurrent_param().active() )
		{
		case GateRecurrentParameter_Active_LINEAR:
			/* do nothing */
			break;
		case GateRecurrentParameter_Active_SIGMOID:
			caffe_cpu_sigmoid_backward( n, data, diff, diff );
			break;
		case GateRecurrentParameter_Active_RELU:
			caffe_cpu_relu_backward( n, data, diff, diff );
			break;
		case GateRecurrentParameter_Active_TANH:
			caffe_cpu_tanh_backward( n, data, diff, diff );
			break;
		default:
			LOG( FATAL ) << "Unknown active method.";
		}
	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::disorder_inputdata( const Dtype * datain, Dtype * dataout, bool slash, bool reverse, int channels )
	{


		const int dims []= {num_,channels,height_,width_};

		const int base_disorder []= {3,2,0,1};  // w,h,n,c
		const int help_order []= {0,1,2,3};  // w,n,h,c
		const int dimsize=4;
		const int helpdims []= {width_,height_,num_,channels};
		
		if(!slash && !reverse_)
		{//  top left to bottom right
			caffe_cpu_permute(datain,dataout,dims,base_disorder,dimsize,-1);
		}
		if(!slash && reverse_)
		{//   top right to bottom left  , flip on width
			caffe_cpu_permute(datain,dataout,dims,base_disorder,dimsize,0);
		}
		if(slash && !reverse_)
		{//  bottom left to top right ,  flip on height
			caffe_cpu_permute(datain,dataout,dims,base_disorder,dimsize,1);
		}
		
		if(slash && reverse_)
		{//   bottom right to top left , flip on width and height
			caffe_cpu_permute(datain,help_disorder_buffer_.mutable_cpu_data(),dims,base_disorder,dimsize,0);
			caffe_cpu_permute(help_disorder_buffer_.mutable_cpu_data(),dataout,helpdims,help_order,dimsize,1);
	
		}

		return;
	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::reorder_outputdata( const Dtype * datain, Dtype * dataout, bool slash, bool reverse, int channels )
	{

		const int recoverdims [] ={width_,height_,num_,channels};
		const int recoverorder []= {2,3,1,0}; //n c h w
		const int help_order []= {0,1,2,3};  
		const int dimsize=4;
		const int helpdims [] = {num_,channels,height_,width_};

		if(!slash && !reverse_)
		{//  top left to bottom right
			caffe_cpu_permute(datain,dataout,recoverdims,recoverorder,dimsize,-1);
		}
		if(!slash && reverse_)
		{//   top right to bottom left  , flip on width
			caffe_cpu_permute(datain,dataout,recoverdims,recoverorder,dimsize,3);
		}
		if(slash && !reverse_)
		{//  bottom left to top right  , flip on height
			caffe_cpu_permute(datain,dataout,recoverdims,recoverorder,dimsize,2);
		}
		
		if(slash && reverse_)
		{//   bottom right to top left , flip on width and height
			caffe_cpu_permute(datain,help_disorder_buffer_.mutable_cpu_data(),recoverdims,recoverorder,dimsize,3);
			caffe_cpu_permute(help_disorder_buffer_.mutable_cpu_data(),dataout,helpdims,help_order,dimsize,2);
		}

		return;
	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*> & bottom,
							    const vector<Blob<Dtype>*> & top )
	{
		CHECK( bottom.size() == 2 ) << "bottom size can only be  2, here get " << bottom.size();
		CHECK( top.size() == 1 ) << "top size must equal to 1";



		num_	= bottom[0]->num();
		height_ = bottom[0]->height();
		width_	= bottom[0]->width();

		channels_	= bottom[0]->channels();
		height_out_	= bottom[0]->height();
		width_out_	= bottom[0]->width();

		num_output_ = this->layer_param_.gaterecurrent_param().num_output();

		slash_		= this->layer_param_.gaterecurrent_param().slash();
		reverse_	= this->layer_param_.gaterecurrent_param().reverse();

		restrict_g_ = this->layer_param_.gaterecurrent_param().restrict_g();


		CHECK( channels_ == num_output_ ) << "we do not support Wx in this version , so channels need equal to num_output, ask liangji or sifei for details";


		this->blobs_.resize( 0 ); /* do not need params */

		CHECK_EQ( num_, bottom[1]->num() ) << "gate num must equal to data num";
		CHECK_EQ( height_, bottom[1]->height() ) << "gate height must equal to data height";
		CHECK_EQ( width_, bottom[1]->width() ) << "gate width must equal to data width";
		CHECK_EQ( num_output_, bottom[1]->channels() ) << "gate channels must equal to data numoutput";


		T_		= width_;
		col_length_	= height_;


		M_ = col_length_;
		N_ = num_ * num_output_;

		/* Shape the tops. */
		for ( int top_id = 0; top_id < top.size(); ++top_id )
		{
			top[top_id]->Reshape( num_, num_output_, height_out_, width_out_ );
		}

		x_disorder_buffer_.Reshape( num_, channels_, height_, width_ );
		h_disorder_buffer_.Reshape( num_, num_output_, height_, width_ );
		help_disorder_buffer_.Reshape( num_, num_output_, height_, width_ );

		
		this->gate_disorder_buffer_.Reshape( num_, num_output_, height_, width_ );
		this->H_t_data_buffer_.Reshape( 1, 1, M_, N_ );
		this->G_t_data_buffer_.Reshape( 1, 1, M_, N_ );
		this->H_t_add1_data_buffer_.Reshape( 1, 1, M_, N_ );
		this->G_t_add1_data_buffer_.Reshape( 1, 1, M_, N_ );
		this->H_t_minus1_data_buffer_.Reshape( 1, 1, M_, N_ );
		this->temp_data_buffer_.Reshape( 1, 1, M_, N_ );
		
	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::Reshape( const vector<Blob<Dtype>*> & bottom,
							 const vector<Blob<Dtype>*> & top )
	{

	
		
		CHECK_EQ( num_, bottom[1]->num() );
		CHECK_EQ( height_, bottom[1]->height() );
		CHECK_EQ( width_, bottom[1]->width() );
		CHECK_EQ( num_output_, bottom[1]->channels() ) << "gate channels must equal to num_output";
		



	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
							     const vector<Blob<Dtype>*> & top )
	{
		const Dtype	* bottom_data	= bottom[0]->cpu_data();
		Dtype		* top_data	= top[0]->mutable_cpu_data();
		Dtype		* X_data	= x_disorder_buffer_.mutable_cpu_data();



		//get x data
 		disorder_inputdata((const Dtype *)bottom_data,X_data,slash_,reverse_,channels_);

		//get gate data
		disorder_inputdata((const Dtype *)bottom[1]->cpu_data(),this->gate_disorder_buffer_.mutable_cpu_data(),slash_,reverse_,num_output_);

		Dtype* G_data = this->gate_disorder_buffer_.mutable_cpu_data();
		Dtype* H_data = this->h_disorder_buffer_.mutable_cpu_data();
		Dtype* temp_data = this->temp_data_buffer_.mutable_cpu_data();
		Dtype* G_t_data = this->G_t_data_buffer_.mutable_cpu_data();
		Dtype* H_t_minus1_data = this->H_t_minus1_data_buffer_.mutable_cpu_data();
	
		const int X_count = M_* N_;
		const int G_count = M_* N_;
		const int H_count = M_* N_;
		const int one_row = num_output_ * num_;

		if(restrict_g_ < 1.0)
			caffe_scal(gate_disorder_buffer_.count(), restrict_g_, G_data);

		for(int t=0; t < T_; t++)
		{
			Dtype* G_t = G_data + t * G_count;
			Dtype* X_t = X_data + t * X_count;
			Dtype* H_t = H_data + t * H_count;
			Dtype* H_t_1 = H_data + (t -1) * H_count;
			Dtype* G_t_1 = G_data + (t -1) * G_count;

			if(t > 0)
			{
				if(reverse_)
				{
					caffe_copy(G_count-one_row,G_t_1 ,G_t_data+one_row); 
					caffe_set(one_row, Dtype(0), G_t_data); 
					caffe_copy(H_count-one_row,H_t_1,H_t_minus1_data + one_row); 
					caffe_set(one_row, Dtype(0), H_t_minus1_data); 
	
				}
				else
				{
					caffe_copy(G_count - one_row,G_t + one_row ,G_t_data +one_row);
					caffe_set(one_row, Dtype(0), G_t_data); 
					caffe_copy(H_count-one_row,H_t_1,H_t_minus1_data + one_row); 
					caffe_set(one_row, Dtype(0),H_t_minus1_data); 
				}
			}
			

			//H(t)=X(t)
			caffe_copy(X_count,X_t,H_t);

			// if t>0
			if(t>0)
			{
				//H(t)-=G(t)*X(t)
				caffe_mul<Dtype>(X_count, G_t_data, X_t, temp_data);
				caffe_sub<Dtype>(X_count, H_t,  temp_data, H_t);
		
				//H(t)+=G(t)*H(t-1)
				caffe_mul<Dtype>(H_count, G_t_data , H_t_minus1_data ,temp_data);
				caffe_add<Dtype>(H_count, H_t,  temp_data, H_t);
			}

		}

		reorder_outputdata((const Dtype *)h_disorder_buffer_.cpu_data(),top_data,slash_,reverse_,num_output_);

	}


	template <typename Dtype>
	void DiagonalGateRecurrentLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
							      const vector<bool> & propagate_down,
							      const vector<Blob<Dtype>*> & bottom )
	{


		const int X_count = M_* N_;
		const int G_count = M_* N_;
		const int H_count = M_* N_;
		const int one_row = num_output_ * num_;

		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype * H_data = h_disorder_buffer_.cpu_data();
		const Dtype * X_data = x_disorder_buffer_.cpu_data();
		const Dtype* G_data = this->gate_disorder_buffer_.cpu_data();


		
	
		
		Dtype* H_diff = h_disorder_buffer_.mutable_cpu_diff();
		Dtype* X_diff = x_disorder_buffer_.mutable_cpu_diff();
		Dtype* G_diff = this->gate_disorder_buffer_.mutable_cpu_diff();

		Dtype* temp_data = this->temp_data_buffer_.mutable_cpu_data();
		Dtype* H_t_minus1_data = this->H_t_minus1_data_buffer_.mutable_cpu_data();
		Dtype* H_t_add1_diff = this->H_t_add1_data_buffer_.mutable_cpu_diff();

		Dtype* G_t_data = this->G_t_data_buffer_.mutable_cpu_data();
		Dtype* G_t_diff = this->G_t_data_buffer_.mutable_cpu_diff();
		Dtype* G_t_add1_data = this->G_t_add1_data_buffer_.mutable_cpu_data();


		//H_diff = top_diff
		disorder_inputdata((const Dtype *)top_diff,H_diff,slash_,reverse_,num_output_);
		
		for(int t= T_ - 1; t >= 0; t--)
		{

			const Dtype* G_t = G_data + t * G_count;
			const Dtype* X_t = X_data + t * X_count;
			const Dtype* H_t = H_data + t * H_count;
			
			Dtype * H_t_diff = H_diff + t*H_count;
			Dtype * X_t_diff = X_diff + t*X_count;
			//Dtype * G_t_diff = G_diff + t*G_count;

			if(t > 0)
			{
				if(reverse_)
				{
					caffe_copy(G_count-one_row,G_data + (t-1) * G_count ,G_t_data+one_row); 
					caffe_set(one_row, Dtype(0), G_t_data); 
					caffe_copy(H_count-one_row,H_data + (t-1) * H_count,H_t_minus1_data + one_row); 
					caffe_set(one_row, Dtype(0),H_t_minus1_data); 

					
	
				}
				else
				{
					caffe_copy(G_count - one_row,G_t + one_row ,G_t_data +one_row);
					caffe_set(one_row, Dtype(0), G_t_data); 
					caffe_copy(H_count-one_row,H_data + (t-1) * H_count,H_t_minus1_data + one_row); 
					caffe_set(one_row, Dtype(0),H_t_minus1_data); 

					 

				}
			}
			
			
			if(t<T_-1)
			{

				if(reverse_)
				{
					caffe_copy(G_count-one_row,G_t,G_t_add1_data); 
					caffe_set(one_row, Dtype(0), G_t_add1_data + G_count - one_row); 
		
					caffe_copy(H_count-one_row,H_diff + (t+1)*H_count + one_row, H_t_add1_diff); 
					caffe_set(one_row, Dtype(0), H_t_add1_diff + H_count - one_row); 
	
				}
				else
				{
					caffe_copy(G_count-one_row,G_data + (t+1) * G_count + one_row,G_t_add1_data); 
					caffe_set(one_row, Dtype(0), G_t_add1_data + G_count - one_row); 

					caffe_copy(H_count-one_row,H_diff + (t+1)*H_count + one_row, H_t_add1_diff); 
					caffe_set(one_row, Dtype(0), H_t_add1_diff + H_count - one_row); 
				}
			}


			//H(t)_diff += H(t+1)_diff * G(t+1)  if t < T-1
			if(t<T_-1)
			{
				caffe_mul<Dtype>(H_count, H_t_add1_diff, G_t_add1_data,temp_data);
				caffe_add<Dtype>(H_count, H_t_diff,  temp_data, H_t_diff);
			}
			//X(t)_diff = H(t)_diff
			caffe_copy(X_count,H_t_diff,X_t_diff);

			//X(t)_diff -= G(t)*H(t)_diff  if t >0
			if(t>0 )
			{
				caffe_mul<Dtype>(H_count, G_t_data,  H_t_diff, temp_data);
				caffe_sub<Dtype>(X_count, X_t_diff,  temp_data, X_t_diff);
			
			}

			//G(t)_diff = H(t)_diff * H(t-1) if t > 0
			if(t>0)
			{
				caffe_mul<Dtype>(H_count,   H_t_diff, H_t_minus1_data, G_t_diff);

				//G(t)_diff -= H(t)_diff*X(t) if t > 0
				caffe_mul<Dtype>(H_count, X_t,  H_t_diff, temp_data);
				caffe_sub<Dtype>(G_count, G_t_diff,  temp_data, G_t_diff);
				
				if(reverse_)
				{
					caffe_copy(G_count - one_row,G_t_diff + one_row,G_diff + (t-1)* G_count);
				}
				else
				{
					caffe_copy(G_count,G_t_diff,G_diff + t * G_count);
				}
			}


			reorder_outputdata((const Dtype *)X_diff,bottom[0]->mutable_cpu_diff(),slash_,reverse_,channels_);
			
			if(restrict_g_ < 1)
                		caffe_scal(gate_disorder_buffer_.count(), restrict_g_, G_diff);
			reorder_outputdata((const Dtype *)G_diff,bottom[1]->mutable_cpu_diff(),slash_,reverse_,num_output_);

		}
	}


#ifdef CPU_ONLY
	STUB_GPU( DiagonalGateRecurrentLayer );
#endif

	INSTANTIATE_CLASS( DiagonalGateRecurrentLayer );
	REGISTER_LAYER_CLASS( DiagonalGateRecurrent );
}  /* namespace caffe */
