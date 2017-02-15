% run test
addpath('../');
addpath('../..');
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
caffe.set_device(1);

save_file = fullfile(state_path, sprintf('helen_%.2d.mat', train_id));
load(save_file);
load(fullfile('helen', 'MultiHelen_v1.mat'));
load(fullfile('helen', 'data_align.mat'));
Solver.batchsize = 1;
Solver.idpool = 1:Solver.testnum;

for id = 1:Solver.testnum
	fprintf('processing the %d th image...\n',id);
	Solver.iter = id;
	[batch, label, ledge] = datalayer_helen_align(Solver, 'test');
	batchc = {single(batch)};
    tic;
	active = net_.forward(batchc);toc
	for c = 1:length(active)
		active_ = active{c};
		if size(active_,3)==11
			res_label = T1_softmax(active_);
			vis_label = vishelen(batch + 0.5, res_label);
			save(fullfile(result_path,sprintf('res_label_%.4d.mat',id)),'res_label');
			imwrite(vis_label, fullfile(result_path,sprintf('label_%.4d.png',id)));
		elseif size(active_,3)==1
			res_edge = 1./(1+exp(-active_));
			save(fullfile(result_path, sprintf('res_edge_%.4d.mat',id)), 'res_edge');
		else
			error('incorrect output channel.');
		end
	end
end
