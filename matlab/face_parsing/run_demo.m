% run_demo
addpath('../');
addpath('../..');
try
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

img = imread('YOUR IMAGE PATH');
img_new = AlignHelen(img, lmk, mean_shape);

[label,edge] = test_1_image_11cls(net_,img_new);
vis_label = vishelen(im2double(img), label);
save(fullfile(result_path,'res_label.mat'),'res_label');
imwrite(vis_label, fullfile(result_path,'vis_label.png'));
