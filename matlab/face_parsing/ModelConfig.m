function [Solver, train] = ModelConfig(model_path, parm)

state_path = fullfile(model_path, sprintf('TrainID_%.2d',parm.train_id));
save_file = fullfile(state_path, sprintf('helen_%.2d.mat', parm.train_id));

solverdef = fullfile(model_path, sprintf('helen_solver_%.2d.prototxt',parm.train_id));
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
    	fprintf('ALERT: no pretrained G found.');
    end
end
Solver = mnist_init(Solver, solverdef);
Solver.matfile = save_file; 
Solver.state_path = state_path; 
end