% 2 resolutions
function [batch_lowres, batch_highres, label_gt] = datalayer_helen_align_2res(Solver, mode)
	batch_lowres = single(zeros(Solver.patchsize, Solver.patchsize, 3, Solver.batchsize));
	batch_highres = single(zeros(Solver.highres, Solver.highres, 3, Solver.batchsize));
	label_gt = single(zeros(Solver.highres, Solver.highres, 11, Solver.batchsize));
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
			img = im2double(imread(Solver.data.trainlist_img{batch_ids(m)}));
			lmk = Solver.data.trainlist_lmk{batch_ids(m)};
			fol = Solver.data.trainlist_lab{batch_ids(m)};
			for k = 1:11
				if k == 1
					lab = im2double(imread(fullfile(fol,sprintf('%s_lbl%.2d',fol,k-1))));
				else
					lab(:,:,k) = im2double(imread(fullfile(fol,sprintf('%s_lbl%.2d',fol,k-1))));
				end
			end
			[img_trans,retform] = AlignHelen(img, lmk, Solver.mean_shape, Solver.highres/250);
			[lab_trans,retform] = AlignHelen(lab, lmk, Solver.mean_shape, Solver.highres/250);
			A = augmentation(size(img_trans,1), size(img_trans,2));
			T = maketform('affine', A);
			simg = single(imtransform(img_trans, T, 'XYScale',1));
			if rand > 0.8 
       			simg = imrandfilter(simg);
   			end
			slb = single(imtransform(lab_trans(:,:,1), T, 'XYScale',1,  'FillValues', 1));
			slb(:,:,2:11) = single(imtransform(lab_trans(:,:,2:11), T, 'XYScale', 1,  'FillValues', 0));
			batch_highres(:,:,:,m) = imresize(simg, [Solver.highres,Solver.highres],'bilinear')-0.5;
			label_gt(:,:,:,m) = imresize(slb, [Solver.highres,Solver.highres], 'nearest');
			batch_lowres(:,:,:,m) = imresize(simg, [Solver.lowres,Solver.lowres],'bilinear')-0.5;
		else
			img = im2double(imread(Solver.data.testlist_img{batch_ids(m)}));
			lmk = Solver.data.testlist_lmk{batch_ids(m)};
			fol = Solver.data.testlist_lab{batch_ids(m)};
			for k = 1:11
				if k == 1
					lab = im2double(imread(fullfile(fol,sprintf('%s_lbl%.2d',fol,k-1))));
				else
					lab(:,:,k) = im2double(imread(fullfile(fol,sprintf('%s_lbl%.2d',fol,k-1))));
				end
			end
			[img_trans,retform] = AlignHelen(img, lmk, Solver.mean_shape, Solver.highres/250);
			[lab_trans,retform] = AlignHelen(lab, lmk, Solver.mean_shape, Solver.highres/250);
			batch_highres(:,:,:,m) = imresize(img_trans, [Solver.highres,Solver.highres],'bilinear')-0.5;
			label_gt(:,:,:,m) = imresize(lab_trans, [Solver.highres,Solver.highres], 'nearest');
			batch_lowres(:,:,:,m) = imresize(img_trans, [Solver.lowres,Solver.lowres],'bilinear')-0.5;
		end
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