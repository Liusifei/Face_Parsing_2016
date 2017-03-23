% set batchsize of RNN as 10*batchsize, then crop 10 patches from each highres map
% batch_highres/label_gt: 1024 * 1024, crop 128 * 128 
% lab_lowres: 128 *128, crop 16 * 16
function [patch_batch, patch_gt, patch_lab] = datalayer_randcrop4rnn(Solver, batch_highres, label_gt, lab_lowres)
	batchsize = 10 * Solver.batchsize;
	patchsize = Solver.patchsize;
	% crop on label_gt
	while true
		rng('shuffle');
		left = ceil(rand * (Solver.highres - patchsize - 1));
		top = ceil(rand * (Solver.highres - patchsize - 1));
		test_lab_1 = label_gt(top+1 : top + patchsize, left+1 : left + patchsize, 2);
		test_lab_2 = label_gt(top+1 : top + patchsize, left+1 : left + patchsize, 11);
		if length(unique(test_lab_1(:))) == 1 && length(unique(test_lab_2(:))) == 1
			continue;
		else
			patch_gt = label_gt(top+1 : top + patchsize, left+1 : left + patchsize, :);
			patch_batch = batch_highres(top+1 : top + patchsize, left+1 : left + patchsize, :);
			left_lowres = floor(left/8); 
			top_lowres = floor(top/8);
			patch_lab = lab_lowres(top_lowres+1:top_lowres+patchsize/8, left_lowres+1:left_lowres+patchsize/8, :);
			break
		end
	end
end