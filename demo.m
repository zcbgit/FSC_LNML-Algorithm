%% BEST RUN WITH MATLAB R2018b!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fuzzy subspace clustering noisy image segmentation algorithm with adaptive local variance 
% & non-local information and mean membership linking
% Engineering Applications of Artificial Intelligence
% This code was solely written by Tongyi Wei.
%
% Detailed membership degrees in a randomly collected 5x5 local area can be seen in the following two matrices:
% U_cluster1_local_5x5
% U_cluster2_local_5x5
%
% Basically, you can run this code SEVERAL times to acquire the most desired result.
% It is welcomed to change the following parameters as you like to see what gonna happen.
%
% CUDA is required in this version. 
% However there is no need to install CUDA seperately since MATLAB has done all the work.
%
% Inputs:
% m - membership factor
% error - Minimum Error
% max_iter - Maximum iterations
% phi - Variance control parameters of Eq.(16)
% tao - Weighting factor
% sigm - Weighted regular parameter
% density - Mixed noise density
% cluster_num - Number of Clustering
% ============== Parameters of non-local spatial information================
% l - Side length of local block
% S - Side length of non-local block
% g - Attenuation of exponential function in Eqs. (10)-(11)
% sigma - Gaussian standard deviation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Intialization
clc
clear
warning off
close all
%% Parameters
m = 2;
error = 0.001;
max_iter = 100;
phi=500;
tao=2;
sigm=1e-5;
density = 0.05;
cluster_num=2;
% Parameters of non-local spatial information
l = 7;
s = 15;
g=10;
sigma = 4;
%% color map
Color_Map=[0.960012728938293,0.867750283531604,0.539798806341941;
    0.00620528530061783,0.456645636880323,0.807705802687295;
    0.16111134052427435,0.0634483119173452,0.321794465524281;
    0.394969976441930,0.662682208014249,0.628325568854317;
    0.0945269059837998,0.794795142853933,0.473068960772494];
%% Input Image 
f_uint8=imread('Synthesis1.tif');
f=double(f_uint8);
figure,imshow(f_uint8),title('Original image');
%% Adding mixing noise
f = f / 255;
f = imnoise(f,'gaussian',0,density);
f = imnoise(f,'salt & pepper',density);
f = imnoise(f,'speckle',density);
f=f*255;
figure,imshow(uint8(f)),title('Noise image');
%% Calculate size
[row,col,depth] = size (f);
N = row * col;
%% Computing non-local spatial information and local variance information
f = gpuArray(f);
non_local_infomation = non_local_information(f, l, s, g, sigma);
local_variance = local_variance(non_local_infomation,phi);
%% Pixel reorganization
all_pixels=gather(reshape(double(f), N ,depth));
all_pixels_xi=gather(reshape(double(local_variance), N ,depth));
%% Calculate difference
difference =20*(mean( mean(all_pixels)-mean(all_pixels_xi))).^2 + eps;    % Eq.(29)
% Constraint alpha for conventional FCM
alpha = 1 ./ difference;    % Eq.(28)
% Constraint beta for local information
beta = difference;  % Eq.(27)
%% Allocate memory space
J=zeros(max_iter,1);
[N,depth]=size(all_pixels);
w=zeros(cluster_num, depth);
distants=zeros(N, cluster_num);
%% Initializing membership
U=rand(N,cluster_num);
U_row_sum=sum(U,2);
U=U./repmat(U_row_sum,[1 cluster_num]);
U_m=U.^m;
%%  Detailed membership degrees in a randomly collected 5x5 local area
U_cluster1_local_5x5=zeros(5,5);
U_cluster2_local_5x5=zeros(5,5);
%% Begin Clustering
for iter=1:max_iter
    % Update Clustering Center using Eq.(39)
    center=((U_m')*(all_pixels*alpha))+((U_m')*(all_pixels_xi*beta));  
    center=center./((sum(U_m))'*ones(1,depth)*(alpha+beta));
    % Update the weight matrix using Eq.(46)
    for k=1:cluster_num
        w(k, :)=sum(repmat(U_m(:,k), 1, depth).*(alpha*(all_pixels-repmat(center(k,:), N, 1)).^2)+repmat(U_m(:,k), 1, depth).*(beta*(all_pixels_xi-repmat(center(k,:), N, 1)).^2))+sigm.*ones(1, depth);  %1*D
    end
    w_up=w.^(-1/(tao-1));  
    w=w_up./repmat(sum(w_up,2), 1, depth); % weight
    % Calculate membership links using Eq.(23)
    membership_linking = repmat(log(sum(repmat(mean(U),[N 1])) + 1) .^ 2, [N 1]);
    % Update the membership matrix using Eq.(37)
    for k=1:cluster_num
        distants(:,k)=sum(repmat(w(k, :).^tao, N, 1).*(alpha*(all_pixels-repmat(center(k, :), N, 1)).^2)+repmat(w(k, :).^tao, N, 1).*(beta*(all_pixels_xi-repmat(center(k, :), N, 1)).^2), 2);
    end
    U_numerator=(distants./membership_linking).^(1/(m-1));
    U=U_numerator.*repmat(sum(1./U_numerator,2),[1,cluster_num]);
    U=1./U;
    U_m=U.^m;
    % Check local membership degrees
    U_reshape1 = reshape(U(:, 1), row, col);
    U_reshape2 = reshape(U(:, 2), row, col);
    U_cluster1_local_5x5(:, :, iter) = U_reshape1(116 :120 , 154 : 158); 
    U_cluster2_local_5x5(:, :, iter) = U_reshape2(116 :120 , 154 : 158);
    % Calculate objective function using Eq.(24)
    J(iter)=sum(sum((U_m.*distants)./membership_linking))+sigm*sum(sum(w.^tao));
    fprintf('Iter %d\n', iter);
    % Convergence condition
    if iter > 1 && abs(J(iter) - J(iter - 1)) <= error
        fprintf('Objective function is converged\n');
        break;
    end
    if iter > 1 && iter == max_iter && abs(J(iter) - J(iter - 1)) > error
        fprintf('Objective function is not converged. Max iteration reached\n');
        break;
    end
end
%% Show Output
[~, cluster_indice] = max(U, [], 2);
cluster_indice=reshape(cluster_indice,[row,col]);
result=label2rgb(cluster_indice,Color_Map);
figure,imshow(result),title('result');
%% Import reference image
load('sy1.mat');
%% Performance index
img = Label_image(f_uint8,cluster_indice);
truth_result=Label_image(f_uint8,double(truth));
Vpc=sum(gather(U).^2,'all')/(row*col)*100;
Vpe=-sum(gather(U).*log(gather(U)+eps),'all')/(row*col)*100;
SA=(length(find((abs(img-truth_result))==0)))/(row*col)*100;
fprintf('Fuzzy partition coefficient Vpc = %.2f%%\n', Vpc);
fprintf('Fuzzy partition entropy Vpe = %.2f%%\n', Vpe);
fprintf('Segmentation Accuracy SA = %.2f%%\n', SA);