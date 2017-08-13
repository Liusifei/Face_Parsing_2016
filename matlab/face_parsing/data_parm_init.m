function Solver = data_parm_init(Solver, parm)
Solver.patchsize = 128;
Solver.batchsize = 40;
Solver.dataroot = '../helen'; % modify to your own helen path
image_list_train = dir(fullfile(Solver.dataroot, 'trainset','*jpg'));
image_list_test = dir(fullfile(Solver.dataroot, 'testset','*jpg'));
Solver.trainnum = length(image_list_train);
Solver.testnum = length(image_list_test);
if exist(fullfile(Solver.dataroot, 'data.mat'),'file')
	load(fullfile(Solver.dataroot, 'data.mat'));
else
	for id = 1:Solver.trainnum
		img = fullfile(Solver.dataroot, 'trainset', image_list_train(id).name);
		short = image_list_train(id).name; short = short(1:end-4);
		lab = fullfile(Solver.dataroot, 'labels', short);
		data.trainlist_img(id) = img;
		data.trainlist_lab(id) = lab;
	end
	for id = 1:Solver.testnum
		img = fullfile(Solver.dataroot, 'testset', image_list_test(id).name);
		short = image_list_test(id).name; short = short(1:end-4);
		lab = fullfile(Solver.dataroot, 'labels', short);
		data.testlist_img(id) = img;
		data.testlist_lab(id) = lab;
	end
	fprintf('saving data structure ...\n');
    save(fullfile(Solver.dataroot, 'data.mat'), 'data');
end
Solver.data = data; 
end