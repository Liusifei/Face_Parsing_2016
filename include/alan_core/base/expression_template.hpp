/*
 * expression_template.hpp
 *
 *  Created on: 2016年4月14日
 *      Author: Alan_Huang
 */

#ifndef EXPRESSION_TEMPLATE_HPP_
#define EXPRESSION_TEMPLATE_HPP_


#ifdef FXNET_XINLINE
	#error "FXNET_XINLINE must not be defined"
#endif

#ifdef _MSC_VER
#define FXNET_FORCE_INLINE __forceinline
#else
#define FXNET_FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifdef __CUDACC__
  #define FXNET_XINLINE FXNET_FORCE_INLINE __device__ __host__
#else
  #define FXNET_XINLINE FXNET_FORCE_INLINE
#endif
#define FXNET_CINLINE FXNET_FORCE_INLINE

namespace fxnet {

template<typename Dtype>
struct Expression{
	FXNET_CINLINE Dtype& self(void){
		return *static_cast<Dtype*>(this);
	}
	FXNET_CINLINE Dtype* ptrself(void){
		return static_cast<Dtype*>(this);
	}
};

template<typename OP, typename Tsrc>
struct UnaryOpExpression : public Expression<UnaryOpExpression<OP, Tsrc> >{
	Tsrc & src_var_;
	explicit UnaryOpExpression(Tsrc& src_var): src_var_(src_var){};
};

template<typename OP, typename Tleft, typename Tright>
struct BinaryOpExpression : public Expression<BinaryOpExpression<OP, Tleft, Tright> >{
	 Tleft & l_var_;
	 Tright & r_var_;
	explicit BinaryOpExpression( Tleft& l_var,
			 Tright& r_var): l_var_(l_var), r_var_(r_var){}
};

template<typename OP, typename TA, typename TB, typename TC>
struct TripleOpExpression : public Expression<TripleOpExpression< OP, TA, TB, TC> >{
	 TA & a_var_;
	 TB & b_var_;
	 TC & c_var_;
	explicit TripleOpExpression( TA& a_var,
			 TB& b_var, TC& c_var): a_var_(a_var), b_var_(b_var), c_var_(c_var){}
};


template<typename OPType>
class Trans{
public:
	FXNET_CINLINE void Do();
};

template< typename OP, typename Tsrc>
class Trans<UnaryOpExpression<OP, Tsrc> >{
public:
	explicit Trans( const UnaryOpExpression<OP, Tsrc>& src): src_(src){
	}
	FXNET_CINLINE void Do(){
		OP::Do(src_.src_var_);
	}
protected:
	UnaryOpExpression<OP, Tsrc> src_;
};

template<typename OP, typename Tleft, typename Tright>
class Trans<BinaryOpExpression<OP, Tleft, Tright> >{
public:
	explicit Trans( const BinaryOpExpression<OP, Tleft, Tright>& src): src_(src){
	}
	FXNET_CINLINE void Do(){
		OP::Do(src_.l_var_, src_.r_var_);
	}
protected:
	BinaryOpExpression<OP, Tleft, Tright> src_;
};

template<typename OP, typename TA, typename TB, typename TC>
class Trans<TripleOpExpression<OP, TA, TB, TC> >{
public:
	explicit Trans( const TripleOpExpression<OP, TA, TB, TC>& src): src_(src){
	}
	FXNET_CINLINE void Do(){
		OP::Do(src_.a_var_, src_.b_var_, src_.c_var_);
	}
protected:
	TripleOpExpression<OP, TA, TB, TC> src_;
};


template<typename OP, typename TA, typename TB, typename TC>
FXNET_CINLINE void fxnet_transform( TA& a, TB& b, TC& c){
	Trans<TripleOpExpression<OP, TA, TB, TC> > temp(
			TripleOpExpression<OP, TA, TB, TC>(a, b,c));
	temp.Do();
}

template<typename OP, typename TA, typename TB>
FXNET_CINLINE void fxnet_transform( TA& a, TB& b){
	Trans<BinaryOpExpression<OP, TA, TB > > temp(
			BinaryOpExpression<OP, TA, TB >(a, b ));
	temp.Do();
}


template<typename OP, typename TA >
FXNET_CINLINE void fxnet_transform( TA& a ){
	Trans<UnaryOpExpression<OP, TA > > temp( (UnaryOpExpression<OP, TA >(a)));
	temp.Do();
}


} // namespace caffe


#endif /* EXPRESSION_TEMPLATE_HPP_ */
