function train_11cls(parm)

model_path = 'model-helen';
[Solver, train] = ModelConfig(model_path, parm);
state_path = Solver.state_path;
tmp = fullfile(state_path,'tmp');
if ~exist(tmp,'dir')
    mkdir(tmp);
end
Solver = data_parm_init(Solver, parm); % TODO
if train==1
    begin = Solver.iter+1;
else
    begin = 1;
end
[~, pr] = fileparts(Solver.snapshot_prefix);

for iter = begin : Solver.max_iter
	Solver.iter = iter;
	Solver.Solver_.set_iter(iter);
	[batch, label] = datalayer_helen(Solver, parm, 'train');
	
end

end
