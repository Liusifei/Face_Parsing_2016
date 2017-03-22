function refine_align_11cls(parm)
model_path = 'model-helen';
[Solver, train] = ModelConfig(model_path, parm);
state_path = Solver.state_path;
tmp = fullfile(state_path,'tmp');
if ~exist(tmp,'dir')
    mkdir(tmp);
end
Solver = data_parm_init4rnn(Solver);
if train==1
    begin = Solver.iter+1;
else
    begin = 1;
end
[~, pr] = fileparts(Solver.snapshot_prefix);

for iter = begin : Solver.max_iter
	
end
end