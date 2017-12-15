function im_a = image_adjustment(img,isGamma,gamma)
% The function performs 3D image adjustment over every slice at Z

im_a = zeros(size(img));
len = length(img(1,1,:));
if isGamma
    for i=1:len
    im_a(:,:,i) = imadjust(img(:,:,i),[],[],gamma);
    end
else
    for i=1:len
        im_a(:,:,i) = imadjust(img(:,:,i));
    end
end
end