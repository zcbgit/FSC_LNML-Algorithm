function enhanceGray = local_variance(f,phi)
% Function for local variance information calculation
% Inputs:
% f - non-local information
% phi - Variance control parameters of Eq.(16)
%----------------------------------

img=uint8(gather(f));
Q_jz = mean2(img); 
k = 3;
len = floor(k/2);
grayPad = padarray(img,[len,len], 'symmetric');
[m, n,d] = size(grayPad);
enhancePad = grayPad;
for c=1:d
    for i = len+1:m-len
        for j = len+1:n - len
            block = grayPad(i-len:i+len,j-len:j+len,c);
            blockMean = mean2(block);
            blockVar = std2(block).^2;
            CG=Q_jz/(blockVar+phi);
            enhancePad(i, j,c) = uint8( (1-CG)*blockMean + CG*(grayPad(i, j,c) - blockMean));
        end
    end
end
enhanceGray = enhancePad(1+len:end-len, 1+len:end-len,:);
enhanceGray=double(enhanceGray);
end