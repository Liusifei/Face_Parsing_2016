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
#include "caffe/layers/gaterecurrent2d_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void GateRecurrent2dLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*> & bottom,
							    const vector<Blob<Dtype>*> & top )
	{
  		horizontal_ = this->layer_param_.gaterecurrent2d_param().horizontal();
		reverse_ = this->layer_param_.gaterecurrent2d_param().reverse();
	}


	template <typename Dtype>
	void GateRecurrent2dLayer<Dtype>::Reshape( const vector<Blob<Dtype>*> & bottom,
							 const vector<Blob<Dtype>*> & top )
	{

		CHECK(top.size()==1)<<"top size must equal to 1";
		int bottomsize=bottom.size();
		CHECK(bottom.size()==5)<<"should be input 5 bottoms :( x, g1, g2, g3, index ). now get "<<bottom.size();
		CHECK(bottom.size()>=3);
		int indexchannels=bottom[bottomsize-1]->channels();
		CHECK(indexchannels == bottomsize - 2)<<"index bottom channels should equal to bottom gate num. "<<indexchannels<<" vs "<<(bottomsize-2); 
		height_  = bottom[0]->height();
		width_ = bottom[0]->width();
		num_ = bottom[0]->num();
		for(int i=0;i<bottomsize;i++)
		{
			CHECK(bottom[i]->num() == num_)<<"all bottom num must equal. "<<bottom[i]->num()<<" vs "<<num_;
			CHECK(bottom[i]->height() == height_)<<"all bottom height must equal. "<<bottom[i]->height()<<" vs "<<height_;
			CHECK(bottom[i]->width() == width_)<<"all bottom width must equal. "<<bottom[i]->height()<<" vs "<<height_;
		}
		channels_ = bottom[0]->channels();
		for(int i=0;i<bottomsize-1;i++)
		{
			CHECK(bottom[i]->channels() == channels_)<<"all data and gate channels must equal. "<<bottom[i]->channels()<<" vs "<<channels_;
		}

		for (int top_id = 0; top_id < top.size(); ++top_id) {
        		top[top_id]->Reshape(num_, channels_, height_, width_);
    		}
		
		H_.Reshape(num_, channels_, height_, width_);

	}


	template <typename Dtype>
	void GateRecurrent2dLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
							     const vector<Blob<Dtype>*> & top )
	{
		this->Forward_gpu(bottom,top);

	}


	template <typename Dtype>
	void GateRecurrent2dLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
							      const vector<bool> & propagate_down,
							      const vector<Blob<Dtype>*> & bottom )
	{

		this->Backward_gpu(top,propagate_down,bottom);
	}


#ifdef CPU_ONLY
	STUB_GPU( GateRecurrent2dLayer );
#endif

	INSTANTIATE_CLASS( GateRecurrent2dLayer );
	REGISTER_LAYER_CLASS( GateRecurrent2d );
}  /* namespace caffe */
