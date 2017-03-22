% run_demo
addpath('../');
addpath('../..');
try
	load('meanshape.mat');
	load('helen_mean_shape.mat');
catch
	error('You need a 5pts mean shape.');
end
model_path = 'model-helen';
result_path = 'vis_results';
if ~exist(result_path,'dir')
	mkdir(result_path);
end

train_id = 1;
state_path = fullfile(model_path, sprintf('TrainID_%.2d',train_id));
trainedmodel = fullfile(state_path, 'face_parsing_v1_iter_20800.caffemodel');
testproto = fullfile(model_path, 'face_parsing_v1_test.prototxt');
net_ = caffe.Net(testproto, trainedmodel,'train');
caffe.set_mode_gpu();
caffe.set_device(3);

%img = imread('YOUR IMAGE PATH');
img = imread('content/56.jpg');
lmk = [204,206,336,239,215,291,185,352,271,383];
[img_new,ret] = AlignHelen(img, lmk, mean_shape);
img_new_s = imresize(img_new,[128,128]);
[label,edge] = test_1_image_11cls(net_,img_new_s);
vis_label = vishelen(im2double(img_new_s), label);
res_label = imtransform(imresize(label,[size(img_new,1),size(img_new,2)]),ret,'XData',[1 size(img,2)],...
'YData',[1 size(img,1)],'XYscale',1, 'Fillvalues',0);
save(fullfile(result_path,'res_label.mat'),'res_label');
imwrite(vis_label, fullfile(result_path,'vis_label.png'));
