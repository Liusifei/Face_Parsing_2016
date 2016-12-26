

import os
import sys

def get_pad(name='',bottom='',top='',pad_h=0,pad_w=0):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Pad'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  pad_param {\n'
	
	ss += '    pad_h:'+str(pad_h)+'\n'
	ss += '    pad_w:'+str(pad_w)+'\n'
	ss+= ' }\n}\n'


 	return ss

def get_resize(name='',bottom='',top='',resize_type='',resize_ratio=1):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Resize'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  resize_param {\n'
	
	ss += '    type:'+resize_type+'\n'
	ss += '    resize_ratio:'+str(resize_ratio)+'\n'
	ss+= ' }\n}\n'


 	return ss

def get_slice(name='',bottom='',top='',slice_point=1,slice_dim=1):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Slice'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	if type(top) == list:
		for t in top:
			ss += '  top:"'+t+'"\n'
	else:
		ss += '  top:"'+top+'"\n'
	
	ss += '  slice_param {\n'
	ss += '    slice_dim:'+str(slice_dim)+'\n'
	if type(slice_point) == list:
		for s in slice_point:
			ss += '    slice_point:'+str(s)+'\n'
	else:
		ss += '    slice_point:'+str(s)+'\n'
	ss+= ' }\n}\n'


 	return ss

def get_crop(name='',bottom='',top='',croptype='',crop_w=0,crop_h=0,point_fix_w=0,point_fix_h=0):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Crop'+'"\n'
	if type(bottom) == list:
		for b in bottom:
			ss += '  bottom:"'+b+'"\n'
	else:
		ss += '  bottom:"'+bottom+'"\n'
	
	ss += '  top:"'+top+'"\n'
	
	ss += '  crop_param {\n'
	ss += '    type:'+croptype+'\n'
	ss += '    crop_w:'+str(crop_w)+'\n'
	ss += '    crop_h:'+str(crop_h)+'\n'
	ss += '    point_fix_w:'+str(point_fix_w)+'\n'
	ss += '    point_fix_h:'+str(point_fix_h)+'\n' 



	ss+= ' }\n}\n'


 	return ss

def get_flatten(name='',bottom='',top=''):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Flatten'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n}\n'
	
 	return ss

def get_l2(name='',bottom='',top=''):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'L2Norm'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n}\n'
	
 	return ss


def get_dropout(name='',bottom='',top='',dropout_ratio=0.5):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Dropout'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	
		
	ss += '  dropout_param {\n'
	ss += '    dropout_ratio:'+str(dropout_ratio)+'\n'+'  }\n}\n'

 	return ss

def get_fc(name='',bottom='',top='',numoutput=1):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'InnerProduct'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  param {\n\
	lr_mult: 1\n \
	decay_mult: 1\n  }\n  param {\n \
	lr_mult: 2\n \
	decay_mult: 0\n  }\n'
		
	ss += '  inner_product_param {\n'
	ss += '    num_output:'+str(numoutput)+'\n'
	
	ss += '    weight_filler {\n\
	type: "xavier"\n\
	std: 0.03\n    }\n    bias_filler {\n\
	type: "constant"\n\
	value: 0\n    }\n  }\n}\n'

 	return ss

def get_eltwise(name='',bottom='',top='',typename='SUM'):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Eltwise'+'"\n'
	for b in bottom:
		ss += '  bottom:"'+b+'"\n'
	ss += '  top:"'+top+'"\n'
	

	
	ss += '  eltwise_param {\n      operation:'+typename+'\n  }\n}\n'
		
 	return ss

def get_concat(name='',bottom='',top='',concat_dim=1):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Concat'+'"\n'
	for b in bottom:
		ss += '  bottom:"'+b+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  concat_param {\n'
	ss += '    concat_dim:'+str(concat_dim)+'\n'
	ss += '  }\n'
	ss+='}\n'

 	return ss

def get_bn(name='',bottom='',top='',use_global_stas=0):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'BatchNorm'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'

	ss += '  param {\n'
	ss += '    lr_mult: 0\n  }\n'
	ss += '  param {\n'
	ss += '    lr_mult: 0\n  }\n'
	ss += '  param {\n'
	ss += '    lr_mult: 0\n  }\n'

	ss += '  batch_norm_param {\n'
	ss += '    use_global_stats:'+str(use_global_stas)+'\n'
	ss += '  }\n'
	ss+='}\n'

 	return ss

def get_power(name='',bottom='',top='',scale=1,shift=0,power=1):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Power'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  power_param {\n'
	ss += '    scale:'+str(scale)+'\n'
	ss += '    shift:'+str(shift)+'\n'
	ss += '    power:'+str(power)+'\n'
	ss += '  }\n'
	ss+='}\n'

 	return ss



def get_domaintransform(name='',bottom='',top=''):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'DomainTransform'+'"\n'
	for b in bottom:
		ss += '  bottom:"'+b+'"\n'
	ss += '  top:"'+top+'"\n'
	ss+='}\n'

 	return ss


def get_pool(name='',bottom='',top='',pooltype='',ksize=2,pad=0,stride=2):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Pooling'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	

	
	ss += '  pooling_param {\n'
	ss += '    pool:'+pooltype+'\n'
	if type(ksize) == list:
		ss += '    kernel_h:'+str(ksize[0])+'\n'
		ss += '    kernel_w:'+str(ksize[1])+'\n'
	else:
		ss += '    kernel_size:'+str(ksize)+'\n'

	if type(pad) == list:
		ss += '    pad_h:'+str(pad[0])+'\n'
		ss += '    pad_w:'+str(pad[1])+'\n'
	else:
		ss += '    pad:'+str(pad)+'\n'
	if type(stride) == list:
		ss += '    stride_h:'+str(stride[0])+'\n'
		ss += '    stride_w:'+str(stride[1])+'\n'
	else:
		ss += '    stride:'+str(stride)+'\n'
	ss += '  }\n}\n'
 	return ss


def get_active(name='',bottom='',top='',typename='PReLU'):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+typename+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	

	if typename == 'PReLU':
		ss += '  prelu_param {\n'
		
	
	
		ss += '      filler {\n\
         type: "gaussian"\n\
         std: 0.03\n      }\n   }\n'
	ss+='}\n'

 	return ss

def get_conv(name='',bottom='',top='',ksize=3,numoutput=1,pad=1,stride=1,paramname_w='',paramname_b=''):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Convolution'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  param {\n'
	if len(paramname_w)>0:
		ss+='    name:"'+paramname_w+'"\n'
	ss += '    lr_mult: 1\n'
	ss += '    decay_mult: 1\n  }\n'

	ss += '  param {\n'
	if len(paramname_b)>0:
		ss+='    name:"'+paramname_b+'"\n'
	ss += '    lr_mult: 2\n'
	ss += '    decay_mult: 0\n  }\n'
		

	ss += '  convolution_param {\n'
	ss += '    num_output:'+str(numoutput)+'\n'
	if type(ksize) == list:
		ss += '    kernel_h:'+str(ksize[0])+'\n'
		ss += '    kernel_w:'+str(ksize[1])+'\n'
	else:
		ss += '    kernel_size:'+str(ksize)+'\n'

	if type(pad) == list:
		ss += '    pad_h:'+str(pad[0])+'\n'
		ss += '    pad_w:'+str(pad[1])+'\n'
	else:
		ss += '    pad:'+str(pad)+'\n'
	if type(stride) == list:
		ss += '    stride_h:'+str(stride[0])+'\n'
		ss += '    stride_w:'+str(stride[1])+'\n'
	else:
		ss += '    stride:'+str(stride)+'\n'


	ss += '    weight_filler {\n\
		type: "xavier"\n\
		std: 0.03\n    }\n    bias_filler {\n\
		type: "constant"\n\
		value: 0\n    }\n  }\n}\n'

 	return ss

def get_deconv(name='',bottom='',top='',ksize=3,numoutput=1,pad=1,stride=1,paramname_w='',paramname_b=''):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'Deconvolution'+'"\n'
	ss += '  bottom:"'+bottom+'"\n'
	ss += '  top:"'+top+'"\n'
	ss += '  param {\n'
	if len(paramname_w)>0:
		ss+='    name:"'+paramname_w+'"\n'
	ss += '    lr_mult: 1\n'
	ss += '    decay_mult: 1\n  }\n'

	ss += '  param {\n'
	if len(paramname_b)>0:
		ss+='    name:"'+paramname_b+'"\n'
	ss += '    lr_mult: 2\n'
	ss += '    decay_mult: 0\n  }\n'
		

	ss += '  convolution_param {\n'
	ss += '    num_output:'+str(numoutput)+'\n'
	if type(ksize) == list:
		ss += '    kernel_h:'+str(ksize[0])+'\n'
		ss += '    kernel_w:'+str(ksize[1])+'\n'
	else:
		ss += '    kernel_size:'+str(ksize)+'\n'

	if type(pad) == list:
		ss += '    pad_h:'+str(pad[0])+'\n'
		ss += '    pad_w:'+str(pad[1])+'\n'
	else:
		ss += '    pad:'+str(pad)+'\n'
	if type(stride) == list:
		ss += '    stride_h:'+str(stride[0])+'\n'
		ss += '    stride_w:'+str(stride[1])+'\n'
	else:
		ss += '    stride:'+str(stride)+'\n'


	ss += '    weight_filler {\n\
		type: "xavier"\n\
		std: 0.03\n    }\n    bias_filler {\n\
		type: "constant"\n\
		value: 0\n    }\n  }\n}\n'

 	return ss

def get_conv_active(name='',bottom='',top='',ksize=3,numoutput=1,pad=1,stride=1,paramname_w='',paramname_b='',active="ReLU"):
	s=get_conv(name=name,bottom=bottom,top=top,ksize=ksize,numoutput=numoutput,pad=pad,stride=stride,paramname_w=paramname_w,paramname_b=paramname_b)
	s+=get_active(name=name+'_active',bottom=top,top=top,typename=active)
	return s

def get_deconv_active(name='',bottom='',top='',ksize=3,numoutput=1,pad=1,stride=1,paramname_w='',paramname_b='',active="ReLU"):
	s=get_deconv(name=name,bottom=bottom,top=top,ksize=ksize,numoutput=numoutput,pad=pad,stride=stride,paramname_w=paramname_w,paramname_b=paramname_b)
	s+=get_active(name=name+'_active',bottom=top,top=top,typename=active)
	return s




def get_sprnn(name='',bottom='',top='',paramname_w='',paramname_b='',horizontal='true',reverse='false',restrict_w=-1,active='LINEAR'):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'SpatialRecurrent'+'"\n'

	if type(bottom) == list:
		ss += '  bottom:"'+bottom[0]+'"\n'
		ss += '  bottom:"'+bottom[1]+'"\n'
	else:
		ss += '  bottom:"'+bottom+'"\n'

	ss += '  top:"'+top+'"\n'
	ss += '  param {\n'
	if len(paramname_w)>0:
		ss+='    name:"'+paramname_w+'"\n'
	ss += '    lr_mult: 1\n'
	ss += '    decay_mult: 1\n  }\n'

	ss += '  param {\n'
	if len(paramname_b)>0:
		ss+='    name:"'+paramname_b+'"\n'
	ss += '    lr_mult: 2\n'
	ss += '    decay_mult: 0\n  }\n'
		

	ss += '  spatialrecurrent_param {\n'
	ss += '    horizontal:'+horizontal+'\n'
	ss += '    reverse:'+reverse+'\n'
	ss += '    restrict_w:'+str(restrict_w)+'\n'
	ss += '    active:'+active+'\n'
	

	ss += '    weight_filler {\n\
		type: "xavier"\n\
		std: 0.03\n    }\n    bias_filler {\n\
		type: "constant"\n\
		value: 0\n    }\n  }\n}\n'

 	return ss


def get_gaternn(name='',bottom='',top='',num_output=1,use_wx='false',use_wh='false',use_bias='false',paramname_wx='',paramname_wh='',paramname_b='',
		horizontal='true',reverse='false',restrict_w=-1,restrict_g=1,use_x_gate='true',use_new_fix='true',active='LINEAR'):

	ss ='layer{\n'
	ss += '  name:"'+name+'"\n'
	ss += '  type:"'+'GateRecurrent'+'"\n'

	if type(bottom) == list:
		ss += '  bottom:"'+bottom[0]+'"\n'
		ss += '  bottom:"'+bottom[1]+'"\n'
	else:
		ss += '  bottom:"'+bottom+'"\n'

	ss += '  top:"'+top+'"\n'

	ss += '  param {\n'
	if len(paramname_wx)>0:
		ss+='    name:"'+paramname_wx+'"\n'
	ss += '    lr_mult: 1\n'
	ss += '    decay_mult: 1\n  }\n'

	ss += '  param {\n'
	if len(paramname_wh)>0:
		ss+='    name:"'+paramname_wh+'"\n'
	ss += '    lr_mult: 1\n'
	ss += '    decay_mult: 1\n  }\n'

	ss += '  param {\n'
	if len(paramname_b)>0:
		ss+='    name:"'+paramname_b+'"\n'
	ss += '    lr_mult: 2\n'
	ss += '    decay_mult: 0\n  }\n'
		

	ss += '  gaterecurrent_param {\n'
	ss += '    num_output:'+str(num_output)+'\n'
	ss += '    horizontal:'+horizontal+'\n'
	ss += '    reverse:'+reverse+'\n'
	ss += '    restrict_w:'+str(restrict_w)+'\n'
	ss += '    active:'+active+'\n'

	ss += '    restrict_g:'+str(restrict_g)+'\n'
	
	ss += '    use_wx:'+use_wx+'\n'
	ss += '    use_wh:'+use_wh+'\n'
	ss += '    use_bias:'+use_bias+'\n'

	ss += '    use_x_gate:'+use_x_gate+'\n'
	ss += '    use_new_fix:'+use_new_fix+'\n'
	
	

	ss += '    weight_filler {\n\
		type: "xavier"\n\
		std: 0.03\n    }\n    bias_filler {\n\
		type: "constant"\n\
		value: 0\n    }\n  }\n}\n'

 	return ss



def get_conv_bn(name='',bottom='',top='',ksize=3,numoutput=1,pad=1,stride=1,paramname_w='',paramname_b='',active="ReLU"):
	s=get_conv(name=name+'_conv',bottom=bottom,top=name+'_conv',ksize=ksize,numoutput=numoutput,pad=pad,stride=stride,paramname_w=paramname_w,paramname_b=paramname_b)
	s+=get_bn(name=name+'_bn',bottom=name+'_conv',top=name+'_bn')
	s+=get_active(name=top,bottom=top+'_bn',top=top,typename=active)
	return s

def get_deconv_bn(name='',bottom='',top='',ksize=3,numoutput=1,pad=1,stride=1,paramname_w='',paramname_b='',active="ReLU"):
	s=get_deconv(name=name+'_deconv',bottom=bottom,top=name+'_deconv',ksize=ksize,numoutput=numoutput,pad=pad,stride=stride,paramname_w=paramname_w,paramname_b=paramname_b)
	s+=get_bn(name=name+'_bn',bottom=name+'_deconv',top=name+'_bn')
	s+=get_active(name=top,bottom=name+'_bn',top=top,typename=active)
	return s

def get_res_unit(name='',bottom='',top='', ch=1,active='PReLU'):
	ss=''
	ss+=get_conv_bn(name=name+'_conv1_1',bottom=bottom,top=name+'_conv1_1',ksize=1,numoutput=ch/2,pad=0,stride=1,active=active)

	ss+=get_conv_bn(name=name+'_conv1_2',bottom=name+'_conv1_1',top=name+'_conv1_2',ksize=3,numoutput=ch,pad=1,stride=1,active=active)

	ss+=get_conv_bn(name=name+'_conv1_3',bottom=name+'_conv1_2',top=name+'_conv1_3',ksize=3,numoutput=ch,pad=1,stride=1,active=active)
	
	ss += get_conv_bn(name=name+'_input',bottom=bottom,top=name+'_input',ksize=1,numoutput=ch,pad=0,stride=1,active=active)

	ss += get_eltwise(name=top,bottom=[name+'_input',name+'_conv1_3'],top=top,typename='SUM')

	return ss
def get_res_unit_stride2(name='',bottom='',top='', ch=1,active='PReLU'):
	ss=''
	ss+=get_conv_bn(name=name+'_conv1_1',bottom=bottom,top=name+'_conv1_1',ksize=1,numoutput=ch,pad=0,stride=1,active=active)

	ss+=get_conv_bn(name=name+'_conv1_2',bottom=name+'_conv1_1',top=name+'_conv1_2',ksize=3,numoutput=ch,pad=1,stride=2,active=active)
	
	ss += get_conv_active(name=name+'_input',bottom=bottom,top=name+'_input',ksize=1,numoutput=ch,pad=0,stride=2,active=active)

	ss += get_eltwise(name=top,bottom=[name+'_input',name+'_conv1_2'],top=top,typename='SUM')

	return ss

def get_res_unit_upsample2(name='',bottom='',top='', ch=1,active='PReLU'):
	ss=''
	ss+=get_deconv_bn(name=name+'_conv1_1',bottom=bottom,top=name+'_conv1_1',ksize=1,numoutput=ch,pad=0,stride=1,active=active)

	ss+=get_deconv_bn(name=name+'_conv1_2',bottom=name+'_conv1_1',top=name+'_conv1_2',ksize=4,numoutput=ch,pad=1,stride=2,active=active)
	
	ss += get_deconv_active(name=name+'_input',bottom=bottom,top=name+'_input',ksize=4,numoutput=ch,pad=1,stride=2,active=active)

	ss += get_eltwise(name=top,bottom=[name+'_input',name+'_conv1_2'],top=top,typename='SUM')

	return ss

   

def get_insec_small(name='',bottom='',top='', ch=1,active='PReLU'):
	ss=''
	ss+=get_conv_active(name=name+'_conv1_1',bottom=bottom,top=name+'_conv1_1',ksize=1,numoutput=ch/2,pad=0,stride=1,active=active)

	ss+=get_conv_active(name=name+'_conv1_2',bottom=name+'_conv1_1',top=name+'_conv1_2',ksize=3,numoutput=ch,pad=1,stride=1,active=active)
	
	ss += get_conv_active(name=name+'_input',bottom=bottom,top=name+'_input',ksize=1,numoutput=ch,pad=0,stride=1,active=active)

	ss += get_eltwise(name=name+'_sum',bottom=[name+'_input',name+'_conv1_2'],top=name+'_sum',typename='SUM')

	ss += get_bn(name=name+'_bn',bottom=name+'_sum',top= top,use_global_stas=0)

	return ss


def get_insec_stride2_small(name='',bottom='',top='', ch=1,active='PReLU'):
	ss=''
	ss+=get_conv_active(name=name+'_conv1_1',bottom=bottom,top=name+'_conv1_1',ksize=1,numoutput=ch,pad=0,stride=1,active=active)

	ss+=get_conv_active(name=name+'_conv1_2',bottom=name+'_conv1_1',top=name+'_conv1_2',ksize=3,numoutput=ch,pad=1,stride=2,active=active)
	
	#ss += get_conv_active(name=name+'_input',bottom=bottom,top=name+'_input',ksize=3,numoutput=ch,pad=1,stride=2,active="ReLU")

	#ss += get_eltwise(name=name+'_sum',bottom=[name+'_input',name+'_conv1_2'],top=name+'_sum',typename='SUM')

	ss += get_bn(name=name+'_bn',bottom=name+'_conv1_2',top= top,use_global_stas=0)

	return ss


def get_insec(name='',bottom='',top='', ch=1):

	ss=''
	ss += get_conv_active(name=name+'_conv1_1',bottom=bottom,top=name+'_conv1_1',ksize=1,numoutput=ch/2,pad=0,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv1_2',bottom=name+'_conv1_1',top=name+'_conv1_2',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")

	
	ss += get_conv_active(name=name+'_conv2_1',bottom=bottom,top=name+'_conv2_1',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv2_2',bottom=name+'_conv2_1',top=name+'_conv2_2',ksize=3,numoutput=ch,pad=1,stride=1,active="ReLU")

	
	ss += get_conv_active(name=name+'_conv3_1',bottom=bottom,top=name+'_conv3_1',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv3_2',bottom=name+'_conv3_1',top=name+'_conv3_2',ksize=3,numoutput=ch/2,pad=1,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv3_3',bottom=name+'_conv3_2',top=name+'_conv3_3',ksize=3,numoutput=ch,pad=1,stride=1,active="ReLU")

	ss += get_conv_active(name=name+'_conv4_1',bottom=bottom,top=name+'_conv4_1',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv4_2',bottom=name+'_conv4_1',top=name+'_conv4_2',ksize=3,numoutput=ch/2,pad=1,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv4_3',bottom=name+'_conv4_2',top=name+'_conv4_3',ksize=3,numoutput=ch/2,pad=1,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv4_4',bottom=name+'_conv4_3',top=name+'_conv4_4',ksize=3,numoutput=ch,pad=1,stride=1,active="ReLU")


	ss += get_concat(name=name+'_concat',bottom=[name+'_conv1_2',name+'_conv2_2',name+'_conv3_3',name+'_conv4_4'],top =name+'_concat')
	ss += get_conv_active(name=name+'_convall',bottom=name+'_concat',top=name+'_convall',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")

	ss += get_conv_active(name=name+'_input',bottom=bottom,top=name+'_input',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")
	
	ss += get_eltwise(name=name+'_sum',bottom=[name+'_input',name+'_convall'],top=name+'_sum',typename='SUM')


	ss += get_bn(name=name+'_bn',bottom=name+'_sum',top= top)

	return ss

def get_insec_stride2(name='',bottom='',top='', ch=1):

	ss=''
	
	ss += get_conv_active(name=name+'_conv2_1',bottom=bottom,top=name+'_conv2_1',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv2_2',bottom=name+'_conv2_1',top=name+'_conv2_2',ksize=3,numoutput=ch,pad=1,stride=2,active="ReLU")

	
	ss += get_conv_active(name=name+'_conv3_1',bottom=bottom,top=name+'_conv3_1',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv3_2',bottom=name+'_conv3_1',top=name+'_conv3_2',ksize=3,numoutput=ch/2,pad=1,stride=1,active="ReLU")
	ss += get_conv_active(name=name+'_conv3_3',bottom=name+'_conv3_2',top=name+'_conv3_3',ksize=3,numoutput=ch,pad=1,stride=2,active="ReLU")

	ss += get_concat(name=name+'_concat',bottom=[name+'_conv2_2',name+'_conv3_3'],top =name+'_concat')
	ss += get_conv_active(name=name+'_convall',bottom=name+'_concat',top=name+'_convall',ksize=1,numoutput=ch,pad=0,stride=1,active="ReLU")

	ss += get_conv_active(name=name+'_input',bottom=bottom,top=name+'_input',ksize=3,numoutput=ch,pad=1,stride=2,active="ReLU")
	
	ss += get_eltwise(name=name+'_sum',bottom=[name+'_input',name+'_convall'],top=name+'_sum',typename='SUM')

	ss += get_bn(name=name+'_bn',bottom=name+'_sum',top= top)

	return ss

def get_INS(name='',bottom='',top='', ch=1):

	ss=''

	ss += get_insec(name=name+'_ins1',bottom=bottom,top=name+'_ins1',ch=ch)

	ss += get_insec(name=name+'_ins2_1',bottom=bottom,top=name+'_ins2_1',ch=ch/2)
	ss += get_insec(name=name+'_ins2_1',bottom=name+'_ins2_1',top=name+'_ins2_2',ch=ch)

	ss += get_insec(name=name+'_ins3_1',bottom=bottom,top=name+'_ins3_1',ch=ch)
	ss += get_insec(name=name+'_ins3_2',bottom=name+'_ins3_1',top=name+'_ins3_2',ch=ch/2)
	ss += get_insec(name=name+'_ins3_3',bottom=name+'_ins3_2',top=name+'_ins3_3',ch=ch)
	
	ss += get_eltwise(name=name+'_sum',bottom=[name+'_ins1',name+'_ins2_2',name+'_ins3_3'],top=name+'_sum',typename='MAX')

	ss += get_bn(name=name+'_bn',bottom=name+'_sum',top= top)

	return ss



def get_MGU(name='',bottom='',top='',seqlength=0,ksize=3,numoutput=1,shareparam=0):
	
	prefix='MGU_'+name+'_'
	ss=''

	name={}
	for i in range(1,seqlength+1):
		name['f'+str(i)]=prefix+'f1'

	x=[]
	h=[]
	f=[]
	ft_convht_1=[]
	ft_convxt=[]
	ft_beforeactive=[]

	ht_hat=[]
	ht_hat_beforeactive=[]
	ht_hat_ftdotht_1=[]
	ht_hat_convftdotht_1=[]
	ht_hat_convxt=[]
	#ht_hat_sum=[]

	ht_1_ft=[]
	ht_1_ft_dotht_1=[]
	ht_ftdotht_hat=[]
	for i in range(0,seqlength+1):
		x.append(prefix+'x'+str(i))
		h.append(prefix+'h'+str(i))
		f.append(prefix+'f'+str(i))
		ft_convht_1.append(prefix+'f'+str(i)+'_convh'+str(i-1))
		ft_convxt.append(prefix + 'f'+str(i)+'_convx'+str(i))
		ht_hat.append(prefix+'h'+str(i)+'_hat')
		ht_hat_ftdotht_1.append(prefix+'h'+str(i)+'_hat_f'+str(i)+'_dot_h'+str(i-1))
		ht_hat_convxt.append(prefix+'h'+str(i)+'_hat_convx'+str(i))
		ht_1_ft.append(prefix+'h'+str(i)+'_1_f'+str(i))
		ht_1_ft_dotht_1.append(prefix+'h'+str(i)+'_1_f'+str(i)+'_dot_h'+str(i-1))
		ht_ftdotht_hat.append(prefix+'h'+str(i)+'_f'+str(i)+'_dot_h'+str(i)+'_hat')
		ft_beforeactive.append(prefix+'f'+str(i)+'_beforeactive')
		ht_hat_beforeactive.append(prefix+'h'+str(i)+'_hat_beforeactive')
		ht_hat_convftdotht_1.append(prefix+'h'+str(i)+'_hat_conv_f'+str(i)+'_dot_h'+str(i-1))
	

	slice_point=[i for i in range(1,seqlength)]

	param_f_h_w=''
	param_f_h_b=''
	param_f_x_w=''
	param_f_x_b=''
	param_hat_h_w=''
	param_hat_h_b=''
	param_hat_x_w=''
	param_hat_x_b=''
	if shareparam:
		param_f_h_w=prefix+'f_h_w'
		param_f_h_b=prefix+'f_h_b'
		param_f_x_w=prefix+'f_x_w'
		param_f_x_b=prefix+'f_x_b'
		param_hat_h_w=prefix+'hat_h_w'
		param_hat_h_b=prefix+'hat_h_b'
		param_hat_x_w=prefix+'hat_x_w'
		param_hat_x_b=prefix+'hat_x_b'

	
	#slice x
	ss += get_slice(name=prefix+'slice',bottom=bottom,top=x[1:],slice_point=slice_point,slice_dim=0)
	
	#get f1 = sigm(conv(x1))
	ss += get_conv_active(name=f[1],bottom=x[1],top=f[1],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_f_x_w,paramname_b=param_f_x_b,active="Sigmoid")

	#get h1_hat = tanh(conv(x1))
	ss += get_conv_active(name=ht_hat[1],bottom=x[1],top=ht_hat[1],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_hat_x_w,paramname_b=param_hat_x_b,active="TanH")

	#get h1 = f1.*h1_hat
	ss += get_eltwise(name=h[1],bottom=[f[1],ht_hat[1]],top=h[1],typename='PROD')
	
	for i in range(2,seqlength+1):
		#get fi = sigm(conv(ht-1) + conv(xt))
		ss += get_conv(name=ft_convht_1[i],top=ft_convht_1[i],bottom=h[i-1],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_f_h_w,paramname_b=param_f_h_b)
		ss += get_conv(name=ft_convxt[i],top=ft_convxt[i],bottom=x[i],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_f_x_w,paramname_b=param_f_x_b)
		ss += get_eltwise(name=ft_beforeactive[i],top=ft_beforeactive[i],bottom=[ft_convht_1[i],ft_convxt[i]],typename='SUM')
		ss += get_active(name=f[i],top=f[i],bottom=ft_beforeactive[i],typename='Sigmoid')
		
		#get hi_hat = tanh(conv(fi.*hi-1) + conv(xi))
		ss += get_eltwise(name=ht_hat_ftdotht_1[i],bottom=[f[i],h[i-1]],top=ht_hat_ftdotht_1[i],typename='PROD')
		ss += get_conv(name=ht_hat_convftdotht_1[i],top=ht_hat_convftdotht_1[i],bottom=ht_hat_ftdotht_1[i],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_hat_h_w,paramname_b=param_hat_h_b)
		ss += get_conv(name=ht_hat_convxt[i],top=ht_hat_convxt[i],bottom=x[i],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_hat_x_w,paramname_b=param_hat_x_b)
		ss += get_eltwise(name=ht_hat_beforeactive[i],top=ht_hat_beforeactive[i],bottom=[ht_hat_convftdotht_1[i],ht_hat_convxt[i]],typename='SUM')
		ss += get_active(name=ht_hat[i],top=ht_hat[i],bottom=ht_hat_beforeactive[i],typename='TanH')

		#get hi = (1-fi).*hi-1 + fi.*hi_hat
		ss += get_power(name=ht_1_ft[i],top=ht_1_ft[i],bottom=f[i],scale=-1,shift=1,power=1)
		ss += get_eltwise(name=ht_1_ft_dotht_1[i],bottom=[ht_1_ft[i],h[i-1]],top=ht_1_ft_dotht_1[i],typename='PROD')
		ss += get_eltwise(name=ht_ftdotht_hat[i],bottom=[f[i],ht_hat[i]],top=ht_ftdotht_hat[i],typename='PROD')
		ss += get_eltwise(name=h[i],bottom=[ht_1_ft_dotht_1[i],ht_ftdotht_hat[i]],top=h[i],typename='SUM')
		

	ss += get_concat(name=top,top=top,bottom =h[1:] ,concat_dim=0)
		#ss += get_conv(name=prefix+'conv_ft_ht-1'+str(i),bottom=top[1],top=prefix+'f1',ksize=ksize,numoutput=numoutput,pad=1,stride=1,paramname_w='',paramname_b='')
	return ss
	




def get_MGU2(name='',bottom='',top='',seqlength=0,ksize=3,numoutput=1,shareparam=0):
	
	prefix='MGU_'+name+'_'
	ss=''

	name={}
	for i in range(1,seqlength+1):
		name['f'+str(i)]=prefix+'f1'

	x=[]
	h=[]
	f=[]
	ft_convht_1=[]
	ft_convxt=[]
	ft_beforeactive=[]

	ht_hat=[]
	ht_hat_beforeactive=[]
	ht_hat_ftdotht_1=[]
	ht_hat_convftdotht_1=[]
	ht_hat_convxt=[]
	#ht_hat_sum=[]

	ht_1_ft=[]
	ht_1_ft_dotht_1=[]
	ht_ftdotht_hat=[]
	for i in range(0,seqlength+1):
		x.append(prefix+'x'+str(i))
		h.append(prefix+'h'+str(i))
		f.append(prefix+'f'+str(i))
		ft_convht_1.append(prefix+'f'+str(i)+'_convh'+str(i-1))
		ft_convxt.append(prefix + 'f'+str(i)+'_convx'+str(i))
		ht_hat.append(prefix+'h'+str(i)+'_hat')
		ht_hat_ftdotht_1.append(prefix+'h'+str(i)+'_hat_f'+str(i)+'_dot_h'+str(i-1))
		ht_hat_convxt.append(prefix+'h'+str(i)+'_hat_convx'+str(i))
		ht_1_ft.append(prefix+'h'+str(i)+'_1_f'+str(i))
		ht_1_ft_dotht_1.append(prefix+'h'+str(i)+'_1_f'+str(i)+'_dot_h'+str(i-1))
		ht_ftdotht_hat.append(prefix+'h'+str(i)+'_f'+str(i)+'_dot_h'+str(i)+'_hat')
		ft_beforeactive.append(prefix+'f'+str(i)+'_beforeactive')
		ht_hat_beforeactive.append(prefix+'h'+str(i)+'_hat_beforeactive')
		ht_hat_convftdotht_1.append(prefix+'h'+str(i)+'_hat_conv_f'+str(i)+'_dot_h'+str(i-1))
	

	slice_point=[i for i in range(1,seqlength)]

	param_f_h_w=''
	param_f_h_b=''
	param_f_x_w=''
	param_f_x_b=''
	param_hat_h_w=''
	param_hat_h_b=''
	param_hat_x_w=''
	param_hat_x_b=''
	if shareparam:
		param_f_h_w=prefix+'f_h_w'
		param_f_h_b=prefix+'f_h_b'
		param_f_x_w=prefix+'f_x_w'
		param_f_x_b=prefix+'f_x_b'
		param_hat_h_w=prefix+'hat_h_w'
		param_hat_h_b=prefix+'hat_h_b'
		param_hat_x_w=prefix+'hat_x_w'
		param_hat_x_b=prefix+'hat_x_b'

	
	#slice x
	ss += get_slice(name=prefix+'slice',bottom=bottom,top=x[1:],slice_point=slice_point,slice_dim=0)
	

	for i in range(1,seqlength+1):
		#get fi = sigm(conv(ht-1) + conv(xt))
		ss += get_conv(name=ft_convht_1[i],top=ft_convht_1[i],bottom=h[i-1],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_f_h_w,paramname_b=param_f_h_b)
		ss += get_conv(name=ft_convxt[i],top=ft_convxt[i],bottom=x[i],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_f_x_w,paramname_b=param_f_x_b)
		ss += get_eltwise(name=ft_beforeactive[i],top=ft_beforeactive[i],bottom=[ft_convht_1[i],ft_convxt[i]],typename='SUM')
		ss += get_active(name=f[i],top=f[i],bottom=ft_beforeactive[i],typename='Sigmoid')
		
		#get hi_hat = tanh(conv(fi.*hi-1) + conv(xi))
		ss += get_eltwise(name=ht_hat_ftdotht_1[i],bottom=[f[i],h[i-1]],top=ht_hat_ftdotht_1[i],typename='PROD')
		ss += get_conv(name=ht_hat_convftdotht_1[i],top=ht_hat_convftdotht_1[i],bottom=ht_hat_ftdotht_1[i],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_hat_h_w,paramname_b=param_hat_h_b)
		ss += get_conv(name=ht_hat_convxt[i],top=ht_hat_convxt[i],bottom=x[i],ksize=ksize,numoutput=numoutput,pad=int(ksize/2),stride=1,paramname_w=param_hat_x_w,paramname_b=param_hat_x_b)
		ss += get_eltwise(name=ht_hat_beforeactive[i],top=ht_hat_beforeactive[i],bottom=[ht_hat_convftdotht_1[i],ht_hat_convxt[i]],typename='SUM')
		ss += get_active(name=ht_hat[i],top=ht_hat[i],bottom=ht_hat_beforeactive[i],typename='TanH')

		#get hi = (1-fi).*hi-1 + fi.*hi_hat
		ss += get_power(name=ht_1_ft[i],top=ht_1_ft[i],bottom=f[i],scale=-1,shift=1,power=1)
		ss += get_eltwise(name=ht_1_ft_dotht_1[i],bottom=[ht_1_ft[i],h[i-1]],top=ht_1_ft_dotht_1[i],typename='PROD')
		ss += get_eltwise(name=ht_ftdotht_hat[i],bottom=[f[i],ht_hat[i]],top=ht_ftdotht_hat[i],typename='PROD')
		ss += get_eltwise(name=h[i],bottom=[ht_1_ft_dotht_1[i],ht_ftdotht_hat[i]],top=h[i],typename='SUM')
		

	ss += get_concat(name=top,top=top,bottom =h[1:] ,concat_dim=0)
		#ss += get_conv(name=prefix+'conv_ft_ht-1'+str(i),bottom=top[1],top=prefix+'f1',ksize=ksize,numoutput=numoutput,pad=1,stride=1,paramname_w='',paramname_b='')
	return ss
	















	







