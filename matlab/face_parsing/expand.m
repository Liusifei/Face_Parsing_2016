function im = expand(im)
    se = strel('square',2);
    im = imdilate(im,se);
    im(im > 1) = 1;
end