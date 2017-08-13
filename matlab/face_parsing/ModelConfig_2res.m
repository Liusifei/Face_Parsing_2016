% Loading the fixed parsing net for lowres result
% Solver with RNN, for refine only
function [Solver, train] = ModelConfig_2res(model_path, parm)
%
state_path = fullfile(model_path, sprintf('TrainRNN_%.2d',parm.train_id));
save_file = fullfile(state_path, sprintf('rnn_%.2d.mat', parm.train_id));
solverdef = fullfile(model_path, sprintf('rnn_solver_%.2d.prototxt',parm.train_id));
if exist(save_file, 'file')
	train = 1;
	M_ = load(save_file);
	fprintf('Loading saved model');
	Solver = T3_SolverParser(solverdef, M_.Solver);
else
	train = 0;
	Solver = T3_SolverParser(solverdef);
end

if isfield(Solver, 'iter')
	[~, pr] = fileparts(Solver.snapshot_prefix);
	state = fullfile(state_path, [pr,sprintf('_iter_%d.solverstate', Solver.iter)]);
	if exist(state,'file') && train
		Solver.state_file = [Solver.snapshot_prefix,sprintf('_iter_%d.solverstate', Solver.iter)];
        Solver.model_file = [Solver.snapshot_prefix, sprintf('_iter_%d.caffemodel', Solver.iter)];
        copyfile(state, Solver.state_file);
        models = fullfile(state_path, [pr,sprintf('_iter_%d.caffemodel', Solver.iter)]);
        copyfile(models, Solver.model_file);
    else
    	delete(fullfile(state_path, [pr, '*']));
    	Solver.state_file = [];
    	fprintf('ALERT: no pretrained model found.');
    end
end
Solver = mnist_init(Solver, solverdef);
Solver.matfile = save_file; 
Solver.state_path = state_path; 

Solver.CNNnet = caffe.Net(parm.cnn_proto, parm.cnn_caffemodel, 'train');
if isfield(parm,'device_id')
	caffe.set_device(parm.device_id);
end
end