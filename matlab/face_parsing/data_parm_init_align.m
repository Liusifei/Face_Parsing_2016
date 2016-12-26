function Solver = data_parm_init_align(Solver)
Solver.patchsize = 128;
Solver.batchsize = 40;
Solver.dataroot = 'helen'; % modify to your own helen path
image_list_train = dir(fullfile(Solver.dataroot, 'trainset','*jpg'));
image_list_test = dir(fullfile(Solver.dataroot, 'testset','*jpg'));
Solver.trainnum = length(image_list_train);
Solver.testnum = length(image_list_test);
load(fullfile(Solver.dataroot,'MultiHelen_v1.mat'));
if exist(fullfile(Solver.dataroot, 'data_align.mat'),'file')
	load(fullfile(Solver.dataroot, 'data_align.mat'));
else
	for id = 1:Solver.trainnum
		img = fullfile(Solver.dataroot, 'AlignedImages_helen', image_list_train(id).name);
		short = image_list_train(id).name; short = short(1:end-4);
		idx = find(strcmp(nameList, short));
		lab = squeeze(lab11(idx,:,:,:));
		data.train(id).impath = img;
		data.train(id).lab = lab;
	end
	for id = 1:Solver.testnum
		img = fullfile(Solver.dataroot, 'AlignedImages_helen', image_list_test(id).name);
		short = image_list_test(id).name; short = short(1:end-4);
		idx = find(strcmp(nameList, short));
		lab = squeeze(lab11(idx,:,:,:));
		data.test(id).impath = img;
		data.test(id).lab = lab;
	end
	fprintf('saving aligned data structure ...\n');
    save(fullfile(Solver.dataroot, 'data_align.mat'), 'data');
end
Solver.data_align = data; 
end
