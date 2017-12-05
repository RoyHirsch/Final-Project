% test 

img1 = loadMRImage('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0010/dataBN.mat',1,exp(-4),0);
loadLabelImage('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0010/gt4.mat',1,97);
figure; imshow(img1(:,:,97));