% no prior
function [batch, label, ledge] = datalayer_helen_align(Solver, mode)
	batch = single(zeros(Solver.patchsize, Solver.patchsize, 3, Solver.batchsize));
	label = single(zeros(Solver.patchsize, Solver.patchsize, 11, Solver.batchsize));
	ledge = single(zeros(Solver.patchsize, Solver.patchsize, 1, Solver.batchsize));
	if strcmp(mode, 'train')
		group = Solver.trainnum / Solver.batchsize;
	else
		group = Solver.testnum / Solver.batchsize;
	end
	idd = mod(Solver.iter, group);
    if idd == 0
    	idd = group;
    end
	batch_ids = Solver.idpool((idd-1)*Solver.batchsize+1:idd*Solver.batchsize);
	for m = 1 : Solver.batchsize
		
		if strcmp(mode, 'train')
			img = im2double(imread(Solver.data_align.train(batch_ids(m)).impath));
			lab = Solver.data_align.train(batch_ids(m)).lab;
			lab = single(lab) / 255;
			A = augmentation(size(img,1), size(img,2));
			T = maketform('affine', A);
			simg = single(imtransform(img, T, 'XYScale',1));
			if rand > 0.8 
       			simg = imrandfilter(simg);
   			end
			slb = single(imtransform(lab(:,:,1), T, 'XYScale',1,  'FillValues', 1));
			slb(:,:,2:11) = single(imtransform(lab(:,:,2:11), T, 'XYScale', 1,  'FillValues', 0));
			batch(:,:,:,m) = imresize(simg, [Solver.patchsize,Solver.patchsize],'bilinear')-0.5;
			label(:,:,:,m) = imresize(slb, [Solver.patchsize,Solver.patchsize], 'nearest');
		else
			img = im2double(imread(Solver.data_align.test(batch_ids(m)).impath));
			lab = Solver.data_align.test(batch_ids(m)).lab;
			batch(:,:,:,m) = imresize(img, [Solver.patchsize,Solver.patchsize],'bilinear')-0.5;
			label(:,:,:,m) = imresize(lab, [Solver.patchsize,Solver.patchsize], 'nearest');
		end
        	ledge(:,:,:,m) = 1-lb2eb1(label(:,:,:,m));
	end
end

function A = augmentation(r, c)
	rng('shuffle');
   rate =  (rand - 0.5)/5;
   shift_x = floor(max(r,c) * rate);
   rate =  (rand - 0.5)/5;
   shift_y = floor(max(r,c) * rate);
   scale_x = 1+(rand-0.5)/5;
   scale_y =  scale_x;
   angle = (rand - 0.5)*(30/180)*pi;
   A = [scale_x * cos(angle), scale_y * sin(angle), shift_x;...
    -scale_x * sin(angle), scale_y * cos(angle), shift_y]';
end
