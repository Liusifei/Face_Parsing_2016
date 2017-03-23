function lmk5 = pts68to5(pts68)
   le_x = mean(pts68(37:42)); le_y = mean(pts68(68+(37:42)));
   re_x = mean(pts68(43:48)); re_y = mean(pts68(68+(43:48)));
   n_x = mean([pts68(31),pts68(34)]); n_y = mean([pts68(68+31),pts68(68+34)]);
   %x = [le_x,re_x,n_x,pts68(49),pts68(55)];
   %y = [le_y,re_y,n_y,pts68(68+49),pts68(68+55)];
   lmk5 = [le_x,le_y,re_x,re_y,n_x,n_y,pts68(49),pts68(68+49),pts68(55),pts68(68+55)]
 end