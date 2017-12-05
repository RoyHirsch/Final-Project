function out = loadLabelImage(dir, isPlot, zSlice)
load(dir)
out = double(gt4);

if isPlot == 1
    figure; imshow(out(:,:,zSlice)/4);
end
end