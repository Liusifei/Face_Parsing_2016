function [IM_shape_new,TFORM] = T1_GeneratingTrom(IM_shape,mean_shape)

% 5 points
base_points = [mean_shape(1) mean_shape(6); mean_shape(2) mean_shape(7);...
    mean_shape(3) mean_shape(8); mean_shape(4) mean_shape(9); mean_shape(5) ...
    mean_shape(10)];
input_points = [IM_shape(1) IM_shape(6); IM_shape(2) IM_shape(7);...
    IM_shape(3) IM_shape(8); IM_shape(4) IM_shape(9); IM_shape(5)...
    IM_shape(10)];

TFORM = cp2tform(input_points, base_points, 'similarity');

[X, Y] = tformfwd(TFORM, IM_shape(1:5), IM_shape(6:10));
IM_shape_new = [X,Y];
IM_shape_new = [X',Y']';
