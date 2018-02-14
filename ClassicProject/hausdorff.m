%% hausdorff = hausdorff(predictClean,label);
% Gets two masks - output is the avarage hausdorff distance
% NOT FINISHED !
%
A = predict;
B = label;
[H, W, D] = size(A);

% 3D edge image
edgeA = zeros(size(A));
edgeB = zeros(size(A));
for z=1:D
    edgeA(:,:,z) = edge(A(:,:,z),'Canny');
    edgeB(:,:,z) = edge(B(:,:,z),'Canny');
end

%% flatten 
flatA = zeros(H*W*D,4);
flatB = zeros(H*W*D,4);
ind = 1;
for i=1:H
    for j=1:W
        for k=1:D
            flatA(ind,:) = [edgeA(i,j,k),i,j,k];
            ind = ind + 1;
        end
    end
end

ind = 1;
for i=1:H
    for j=1:W
        for k=1:D
            flatB(ind,:) = [edgeB(i,j,k),i,j,k];
            ind = ind + 1;
        end
    end
end            
%% 

% filer out zeros and make any array contain only cordinates
flatA = flatA(all(flatA,2),:);
flatB = flatB(all(flatB,2),:);
flatA = flatA(:,2:4);
flatB = flatB(:,2:4);

[LA WA] = size(flatA); 
[LB WB] = size(flatB);
minDistArray = zeros(LA,1);
%
for i=1:LA
    minDist = exp(5);  % initial big number
    for j=1:LB
        pointA = flatA(i,:);
        pointB = flatB(i,:);
        dist = sqrt((pointA(1)-pointB(1))^2 + (pointA(2)-pointB(2))^2 + (pointA(3)-pointB(3))^2);
        if dist < minDist
            minDist = dist;
        end
    end
    minDistArray(i) = minDist;
end
%
% avarageA2Bdist = mean(minDistArray)
