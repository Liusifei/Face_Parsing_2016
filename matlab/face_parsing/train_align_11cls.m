function train_align_11cls(parm)

model_path = 'model-helen';
[Solver, train] = ModelConfig(model_path, parm);
state_path = Solver.state_path;
tmp = fullfile(state_path,'tmp');
if ~exist(tmp,'dir')
    mkdir(tmp);
end
Solver = data_parm_init_align(Solver); % TODO
if train==1
    begin = Solver.iter+1;
else
    begin = 1;
end
[~, pr] = fileparts(Solver.snapshot_prefix);

for iter = begin : Solver.max_iter
	Solver.iter = iter;
	Solver.Solver_.set_iter(iter);
	if ~mod(iter-1, 50)
		rng('shuffle');
		Solver.idpool = randperm(Solver.trainnum);
	end
	[batch, label, ledge] = datalayer_helen_align(Solver, 'train');
	batchc = {single(batch)};
	tic;active = Solver.Solver_.net.forward(batchc);toc
	delta = cell(size(active));
	for c = 1 : length(active)
		active_ = active{c};
		delta{c} = zeros(size(active_));
        if size(active_,3) == 1
		    if iter < parm.change2uni
              [delta_, loss] = Face01_L2_eb(active_, ledge, 'train');
            else
              [delta_, loss] = Face02_L2_eb(active_, ledge, 'train');
            end
            Solver.loss_eb(iter) = loss;
		elseif size(active_,3) == 11
		    if iter < parm.change2uni
                [delta_, loss] = Face01_L2_lb(active_, label, 'train');
            else
                [delta_, loss] = Face02_L2_lb(active_, label, 'train');
            end
            Solver.loss_lb(iter) = loss;
		else
			error('pls check the output channels');
		end
		delta{c} = delta_;
	end
    % DEBUG
    if iter == begin
        save debug.mat delta active label ledge
    end
	if ~isnan(Solver.loss_lb(iter)) || ~isnan(Solver.loss_eb(iter)) 
        f = Solver.Solver_.net.backward(delta);
        Solver.Solver_.update();
    else
        error('NAN');
    end

    if ~mod(iter, 10)
        fprintf('loss_lb: %d, loss_eb: %d\n', mean(Solver.loss_lb(iter-9:iter)), ...
            mean(Solver.loss_eb(iter-9:iter)));
            % vis
        %vis = vishelen(batch+0.5, active_);
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
