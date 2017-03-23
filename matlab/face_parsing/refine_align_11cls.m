function Solver = refine_align_11cls(parm)
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
	[batch_lowres, batch_highres, label_gt] = datalayer_helen_align_2res(Solver, 'train'); % TODO
	% CNN base segmentation
	active = Solver.CNNnet.forward({single(batch_lowres)});
	for c = 1 : length(active)
		active_ = active{c};
        if size(active_,3) == 1
        	% do noting
		elseif size(active_,3) == 11
			lab_lowres = active_; % 128*128 map
		else
			error('pls check the output channels');
		end
	end
	% RNN refine, one output only
	[patch_batch, patch_gt, patch_lab] = ...
		datalayer_randcrop4rnn(Solver, batch_highres, label_gt, lab_lowres); % TODO
	active = Solver.Solver_.net.forward({cat(3,single(patch_batch),single(patch_lab))});
	if length(active) > 2
		error('RNN net should has only one output branch.');
	end
	active_ = active{1}; % a refined patch_lab
	delta = cell(size(active));
    if iter < parm.change2uni
        [delta_, loss] = Face01_L2_lb(active_, patch_gt, 'train');
    else
        [delta_, loss] = Face02_L2_lb(active_, patch_gt, 'train');
    end
    Solver.loss(iter) = loss;
    delta{1} = delta_;
    if ~isnan(Solver.loss(iter))
        f = Solver.Solver_.net.backward(delta);
        Solver.Solver_.update();
    else
        error('NAN');
    end

    if ~mod(iter, 10)
        fprintf('loss: %d\n', mean(Solver.loss(iter-9:iter)));
        save(fullfile(tmp,'act.mat'),'active');
    end
    if mod(iter, 100) == 0
    	Solver.Solver_.save();
    	sys_path = [Solver.snapshot_prefix, sprintf('_iter_%d.solverstate',iter)];
        disp(movefile(sys_path, state_path));
        sys_path = [Solver.snapshot_prefix, sprintf('_iter_%d.caffemodel',iter)];
        disp(movefile(sys_path, state_path));
        save(Solver.matfile, 'Solver');
        data_ = Solver.Solver_.net.get_data();
        diff_ = Solver.Solver_.net.get_diff();
        save(fullfile(tmp,'blob.mat'),'data_','diff_');
    end
    % clean history
	if iter >= 300
        if exist(fullfile(state_path, [pr,sprintf('_iter_%d.solverstate', iter-200)]),'file')
            delete(fullfile(state_path, [pr,sprintf('_iter_%d.solverstate', iter-200)]));
        end
        if exist(fullfile(state_path, [pr,sprintf('_iter_%d.caffemodel', iter-200)]),'file')
            delete(fullfile(state_path, [pr,sprintf('_iter_%d.caffemodel', iter-200)]));
        end
    end    
end
end