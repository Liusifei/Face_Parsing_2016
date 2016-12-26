% run_demo.m
function [label,edge] = test_1_image_11cls(net_,img)

img = im2double(imresize(img,[128,128]));
batchc = {single(img - 0.5)};
active = net_.forward(batchc);
for c = 1:length(active)
	active_ = active{c};
	if size(active_,3)==11
		label = T1_softmax(active_);	
	elseif size(active_,3)==1
		edge = 1./(1+exp(-active_));		
	else
		error('incorrect output channel.');
	end
end
