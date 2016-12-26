/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"
#include "time.h"

namespace caffe {
template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    
    CHECK(bottom[0]->height()== 1);
    CHECK(bottom[0]->width()== 1);
    CHECK(bottom[1]->channels()== 1);
    CHECK(bottom[1]->height()== 1);
    CHECK(bottom[1]->width()== 1);
    CHECK(bottom[1]->num() == bottom[0]->num());

    diff_ap_.Reshape(1, bottom[0]->channels(), 1, 1);
    diff_an_.Reshape(1, bottom[0]->channels(), 1, 1);
    diff_pn_.Reshape(1, bottom[0]->channels(), 1, 1);
    loss_.Reshape(bottom[0]->num()*bottom[0]->num(), 1, 1, 1);
    valid_loss_.Reshape(bottom[0]->num()*bottom[0]->num(), 1, 1, 1);

    
    samplecount_.Reshape(bottom[0]->num(), 1, 1, 1);

    caffe_set(samplecount_.count(),Dtype(0.0),samplecount_.mutable_cpu_data());
    
}
    


template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

    CHECK(bottom[0]->height()== 1);
    CHECK(bottom[0]->width()== 1);
    CHECK(bottom[1]->channels()== 1);
    CHECK(bottom[1]->height()== 1);
    CHECK(bottom[1]->width()== 1);
    CHECK(bottom[1]->num() == bottom[0]->num());

    diff_ap_.Reshape(1, bottom[0]->channels(), 1, 1);
    diff_an_.Reshape(1, bottom[0]->channels(), 1, 1);
    diff_pn_.Reshape(1, bottom[0]->channels(), 1, 1);
    loss_.Reshape(bottom[0]->num()*bottom[0]->num(), 1, 1, 1);


    samplecount_.Reshape(bottom[0]->num(), 1, 1, 1);

    caffe_set(samplecount_.count(),Dtype(0.0),samplecount_.mutable_cpu_data());
  
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {


	double dur;
	clock_t start,end;
	start = clock();
	
	


	int num = bottom[0]->num();
	int datacount = bottom[0]->count() / bottom[0]->num();;

	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
		
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype * diff_ap = diff_ap_.mutable_cpu_data();
	Dtype * diff_an = diff_an_.mutable_cpu_data();
	Dtype * diff_pn = diff_pn_.mutable_cpu_data();
	Dtype * samplecount = samplecount_.mutable_cpu_data();

	Dtype la,lp,ln;
	Dtype margin = this->layer_param_.triplet_loss_param().margin();// alpha
	Dtype dist_sq_ap, dist_sq_an;
	Dtype loss =0.0,total_triplet_count=0.0, valid_triplet_count=0.0;
	const Dtype lossweight =  top[0]->cpu_diff()[0];
	bool seq_bottom = this->layer_param_.triplet_loss_param().sequencial_bottom();

	caffe_set(bottom[0]->count(),Dtype(0.0),bottom_diff);
	caffe_set(samplecount_.count(),Dtype(0.0),samplecount);
	
	
	
	for(int i=0;i<num;i++)
	{
		la = label[i]; //a
		for(int j=i+1;j<num;j++)
		{
			lp = label[j]; //p
			if(seq_bottom)
			{
				if(la!=lp)
					break;
			}
			for(int k=0;k<num;k++)
			{	
				if(k == i || k ==j)
					continue;

				ln =label[k];  //n
				
				if((int)la == (int)lp && (int)la != (int)ln)
				{//find a triplet
					int ida = i, idp = j, idn = k;
					for(int swap=0;swap < 2;swap++)
					{
						int temp = ida;
						ida = idp;
						idp = temp;

						const Dtype * a = bottom_data + ida*datacount;
						const Dtype * p = bottom_data + idp*datacount;
						const Dtype * n = bottom_data + idn*datacount;
						Dtype * a_diff = bottom_diff + ida*datacount;
						Dtype * p_diff = bottom_diff + idp*datacount;
						Dtype * n_diff = bottom_diff + idn*datacount;
					
					
					
						caffe_sub(
							datacount, 
							a, // a
							p, //p
							diff_ap); // diff_ap_= a - p
						caffe_sub(
							datacount,
							a, //a
							n, //n
							diff_an); // diff_an_ = a - n
						caffe_sub(
							datacount, 
							p, //p
							n, //n
							diff_pn // diff_pn_ = p - n
							);

						dist_sq_ap = caffe_cpu_dot(datacount, diff_ap, diff_ap);
						dist_sq_an = caffe_cpu_dot(datacount, diff_an, diff_an);
						Dtype mdist =  std::max(margin + dist_sq_ap- dist_sq_an, Dtype(0.0));

						loss += mdist;
						total_triplet_count +=1;

						if(mdist > 0)
						{// calculate diff
							valid_triplet_count +=1;
							samplecount[ida]+=1;
							samplecount[idp]+=1;
							samplecount[idn]+=1;

							caffe_cpu_axpby(datacount, -1*lossweight, diff_pn, Dtype(1.0), a_diff);
							caffe_cpu_axpby(datacount, -1*lossweight, diff_ap, Dtype(1.0), p_diff);
							caffe_cpu_axpby(datacount, lossweight, diff_an, Dtype(1.0), n_diff); 		     
						}


					}//end swap a p

				}//end find one triplet
			}//end k
		}//end j
	}//end i
 	if(total_triplet_count>0) 
		loss = loss/total_triplet_count/Dtype(2);
	else
		loss = 0;
	top[0]->mutable_cpu_data()[0] = loss;

	if(this->layer_param_.triplet_loss_param().norm_diff() && valid_triplet_count>0)
	{	
		caffe_scal(bottom[0]->count(),Dtype(1.0/valid_triplet_count),bottom[0]->mutable_cpu_diff());
		LOG(INFO)<<"valid loss count = "<<(int)valid_triplet_count<<"; total loss count = "<<total_triplet_count<<"; ratio = "<<valid_triplet_count/total_triplet_count;
	}

	end = clock();
	dur = (double)(end - start);
	if(this->layer_param_.triplet_loss_param().print_time())
		LOG(INFO)<<"TripletLossLaye CPU Use Time: "<<(dur/CLOCKS_PER_SEC);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	;	
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);


}  // namespace caffe
