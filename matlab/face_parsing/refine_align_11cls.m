function refine_align_11cls(parm)
model_path = 'model-helen';
[Solver, train] = ModelConfig_2res(model_path, parm);
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
	Solver.iter = iter;
	Solver.Solver_.set_iter(iter);
	if ~mod(iter-1,50)
		rng('shuffle');
		Solver.idpool = randperm(Solver.trainnum);
	end
	[batch_lowres, batch_highres, label] = datalayer_helen_align_2res(Solver, 'train'); % TODO
	% CNN base segmentation
	act_lowres = Solver.CNNnet.forward({single(batch_lowres)});
	for c = 1 : length(active)
		active_ = active{c};
        if size(active_,3) == 1
        	% do noting
		elseif size(active_,3) == 11
			
		else
			error('pls check the output channels');
		end
		delta{c} = delta_;
	end
end
end