/*
 * Author: Liangji
 * Email: liangji20040249@gmail.com
 */
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/learncrf_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
	template <typename Dtype>
	void LearnCRFLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*> & bottom,
					       const vector<Blob<Dtype>*> & top )
	{
		CHECK( bottom.size() == 2 ) << "must have 2 bottom";
		CHECK( bottom[0]->num() == bottom[1]->num() ) << "should be save num";
		CHECK( bottom[0]->height() == bottom[1]->height() ) << "should be same size.";
		CHECK( bottom[0]->width() == bottom[1]->width() ) << "should be same size.";
		kernel_size_ = this->layer_param_.learn_crf_param().kernel_size();
		CHECK( kernel_size_ * kernel_size_ == bottom[1]->channels() ) << " ask liangji for details";
		pad_ = int( (kernel_size_ - 1) / 2);
		CHECK( pad_ * 2 + 1 == kernel_size_ );
		dilation_ = this->layer_param_.learn_crf_param().dilation();
	}


	template <typename Dtype>
	void LearnCRFLayer<Dtype>::Reshape( const vector<Blob<Dtype>*> & bottom,
					    const vector<Blob<Dtype>*> & top )
	{
		top[0]->ReshapeLike( *bottom[0] );
		height_		= bottom[0]->height();
		width_		= bottom[0]->width();
		channels_	= bottom[0]->channels();
		num_		= bottom[0]->num();

		x_col_buffer_.Reshape( 1, 1, 1, kernel_size_ * kernel_size_ * channels_ * height_ * width_ );
		mul_buffer_.Reshape( 1, 1, 1, kernel_size_ * kernel_size_  );
		
	}


	template <typename Dtype>
	void LearnCRFLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
						const vector<Blob<Dtype>*> & top )
	{
		const Dtype	* bottom_data	= bottom[0]->cpu_data();
		const Dtype	* weight_data	= bottom[1]->cpu_data();

		Dtype * top_data = top[0]->mutable_cpu_data();

		const int	count	= height_ * width_ * channels_;
		const int	w_count = bottom[1]->channels() * bottom[1]->height() * bottom[1]->width();
		const int spdim = height_ * width_;
		Dtype		* x	= x_col_buffer_.mutable_cpu_data();

		caffe_set( mul_buffer_.count(), static_cast<Dtype>( 1 ), mul_buffer_.mutable_cpu_data() );
		const Dtype * mul_data = mul_buffer_.cpu_data();


		CHECK( x_col_buffer_.count() == channels_ * w_count );


		for ( int i = 0; i < num_; i++ )
		{
			const Dtype	* bi	= bottom_data + i * count;
			const Dtype	* wi	= weight_data + i * w_count;
			Dtype		* ti	= top_data + i * count;

//SaveArray("bi.txt", bi,count);

			im2col_cpu( bi, channels_, height_, width_, kernel_size_, kernel_size_, pad_, pad_, 1, 1, dilation_, dilation_, x );
//SaveArray("x.txt", x,w_count * channels_);

			for ( int j = 0; j < channels_; j++ )
			{
				caffe_mul( w_count, wi, x + j * w_count, x + j * w_count );
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, spdim, kernel_size_ * kernel_size_,(Dtype)1.,mul_data, x+j*w_count, (Dtype)0., ti + j*spdim);
			}
//SaveArray("x_muled.txt", x,w_count * channels_);
//SaveArray("wi.txt", wi,w_count);

			

			//col2im_cpu( x, channels_, height_, width_, kernel_size_, kernel_size_, pad_, pad_, 1, 1, dilation_, dilation_, ti );

//SaveArray("ti.txt", ti,count);
//LOG(FATAL);
		}
	}


	template <typename Dtype>
	void LearnCRFLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
						 const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom )
	{
		const Dtype	* bottom_data	= bottom[0]->cpu_data();
		const Dtype	* weight_data	= bottom[1]->cpu_data();
		//const Dtype	* top_data	= top[0]->cpu_data();
		const Dtype	* top_diff	= top[0]->cpu_diff();
		Dtype		* bottom_diff	= bottom[0]->mutable_cpu_diff();

		Dtype * w_diff = bottom[1]->mutable_cpu_diff();

		Dtype	* x		= x_col_buffer_.mutable_cpu_data();
		Dtype	* x_diff	= x_col_buffer_.mutable_cpu_diff();
		const int spdim = height_ * width_;
		const int	count	= height_ * width_ * channels_;
		const int	w_count = bottom[1]->channels() * bottom[1]->height() * bottom[1]->width();
		const Dtype * mul_data = mul_buffer_.cpu_data();

		if ( propagate_down[0] )
		{
			caffe_set( bottom[0]->count(), static_cast<Dtype>( 0 ), bottom_diff );
			caffe_set( bottom[1]->count(), static_cast<Dtype>( 0 ), w_diff );

			for ( int i = 0; i < num_; i++ )
			{
				const Dtype	* bi		= bottom_data + i * count;
				Dtype		* bi_diff	= bottom_diff + i * count;
				const Dtype	* wi		= weight_data + i * w_count;
				Dtype * wi_diff = w_diff + i * w_count;

				
				const Dtype	* ti_diff	= top_diff + i * count;

//SaveArray("bi.txt", bi,count);
				im2col_cpu( bi, channels_, height_, width_, kernel_size_, kernel_size_, pad_, pad_, 1, 1, dilation_, dilation_, x );
//SaveArray("x.txt", x,w_count * channels_);
//SaveArray("ti_diff.txt", ti_diff,count);
				//im2col_cpu( ti_diff, channels_, height_, width_, kernel_size_, kernel_size_, pad_, pad_, 1, 1, dilation_, dilation_, x_diff );
				for(int j=0;j<channels_;j++)
				{
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, kernel_size_ * kernel_size_, spdim, 1,(Dtype)1.,
								mul_data, ti_diff+j*spdim, (Dtype)0., x_diff + j*w_count);
				}
//SaveArray("x_diff.txt", x_diff,channels_ * w_count);
				
				caffe_mul( w_count * channels_, x_diff, x, x );
				for ( int j = 0; j < channels_; j++ )
				{
					caffe_add( w_count, x + j * w_count, wi_diff, wi_diff );
				}
//SaveArray("widiff.txt", wi_diff,w_count);
//SaveArray("wi.txt", wi,w_count);
				for ( int j = 0; j < channels_; j++ )
				{
					caffe_mul( w_count, x_diff+j*w_count, wi, x_diff +j*w_count);
					//caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, spdim, kernel_size_ * kernel_size_,(Dtype)1.,mul_data, x_diff+j*w_count, (Dtype)0., bi_diff + j*spdim);

				}


				col2im_cpu( x_diff, channels_, height_, width_, kernel_size_, kernel_size_, pad_, pad_, 1, 1, dilation_, dilation_, bi_diff );

//SaveArray("bidiff.txt", bi_diff,count);
//LOG(FATAL);

			}
		}
	}


	template <typename Dtype>
	void LearnCRFLayer<Dtype>::Forward_gpu( const vector<Blob<Dtype>*> & bottom,
						const vector<Blob<Dtype>*> & top )
	{
		


		this->Forward_cpu( bottom, top );
	}


	template <typename Dtype>
	void LearnCRFLayer<Dtype>::Backward_gpu( const vector<Blob<Dtype>*> & top,
						 const vector<bool> & propagate_down, const vector<Blob<Dtype>*> & bottom )
	{
		this->Backward_cpu( top, propagate_down, bottom );
	}


#ifdef CPU_ONLY
	STUB_GPU( LearnCRFLayer );
#endif

	INSTANTIATE_CLASS( LearnCRFLayer );
	REGISTER_LAYER_CLASS( LearnCRF );
}  /* namespace caffe */
