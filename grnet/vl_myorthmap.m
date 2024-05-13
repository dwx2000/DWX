function [Y, R] = vl_myorthmap (R, p, dzdy)
%orthonormal mapping (OrthMap) layer
% vl_myorthmap: OrthMap（正交映射）层
%
% 用法：
%   [Y, R] = vl_myorthmap (R, p, dzdy)
%
% 输入参数：
%   - R: 存储中间结果的结构体
%   - p: 正交映射维度
%   - dzdy: 反向传播的导数，对于测试模式可省略
%
% 输出参数：
%   - Y: 输出数据，经过正交映射处理后的结果
%   - R: 存储中间结果的结构体，更新了正交映射相关信息
%
% 详细解释：
%   vl_myorthmap 函数对输入数据进行正交映射处理，计算输出数据 Y。如果存储中间结果
%   为空，则对输入数据 R.x 进行正交映射并返回结果 Y；如果存储中间结果不为空，则
%   根据存储中间结果对输入数据进行反向传播计算，并返回 Y。


X = R.x;  % 获取输入数据
A = R.aux; % 获取存储中间结果的结构体
[n1,n2,n3,n4] = size(X); % 获取输入数据的维度

if isempty(A) == 1
    % 如果存储中间结果为空，则进行前向传播计算
    Y = zeros(n1,p,n3,n4); % 初始化输出 Y

    Us = zeros(n1,n2,n3,n4); % 初始化左奇异矩阵 U
    Ss = zeros(n1,n2,n3,n4); 
%     parfor i3 = 1  : n3
    for i3 = 1  : n3
        for i4 = 1 : n4
                X_t = X(:,:,i3,i4); % 获取当前数据
                [U_t, S_t, V_t] = svd(X_t); % 对当前数据进行奇异值分解
                Us(:,:,i3,i4) = U_t; % 存储左奇异矩阵 U
                Ss(:,:,i3,i4) = S_t; % 存储奇异值矩阵 S
                Y(:,:,i3,i4) = U_t(:,1:p);   % 计算正交映射并更新 Y        
        end
    end
    R.aux{1} = Us; % 更新存储中间结果的结构体
    R.aux{2} = Ss; % 更新存储中间结果的结构体
else
    % 如果存储中间结果不为空，则进行反向传播计算
    Us = A{1}; % 获取左奇异矩阵 U
    Ss = A{2};
    D = size(Ss,2); % 获取奇异值矩阵的维度
    Y = zeros(n1,n2,n3,n4); % 初始化输出 Y

%     parfor i3 = 1  : n3
    for i3 = 1  : n3

        for i4 = 1 : n4
                U_t = Us(:,:,i3,i4); S_t = Ss(:,:,i3,i4);
                % 计算反向传播梯度并更新 Y
                Y(:,:,i3,i4) = calculate_grad_svd(U_t,S_t,p, D,dzdy(:,:,i3,i4));
        end
    end
end



function dzdx = calculate_grad_svd(U,S,p,D,dzdy)
% calculate_grad_svd: 计算奇异值分解的梯度
%
% 用法：
%   dzdx = calculate_grad_svd(U,S,p,D,dzdy)
%
% 输入参数：
%   - U: 左奇异矩阵
%   - S: 奇异值矩阵
%   - p: 正交映射维度
%   - D: 奇异值矩阵的维度
%   - dzdy: 反向传播的导数
%
% 输出参数：
%   - dzdx: 奇异值分解的梯度
%
% 详细解释：
%   calculate_grad_svd 函数根据输入的左奇异矩阵 U、奇异值矩阵 S、正交映射维度 p、
%   奇异值矩阵的维度 D 和反向传播的导数 dzdy，计算奇异值分解的梯度 dzdx。

diagS = diag(S);  % 获取奇异值矩阵的对角线元素
Dmin = length(diagS); % 获取奇异值矩阵的维度
ind = 1:Dmin; % 获取索引范围

dLdC = zeros(D,D); % 初始化导数矩阵
A = [ones(D,p) zeros(D,D-p)];   %创建辅助矩阵 A
dLdC(logical(A)) = dzdy; % 根据导数更新导数矩阵
dLdU = dLdC; % 初始化左奇异矩阵的导数

% 处理奇异值矩阵维度为 1 的情况
if sum(ind) == 1 % diag behaves badly when there is only 1d
    K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))');
    K(eye(size(K,1))>0)=0;
else
    % 计算 K 矩阵
    K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); 
    % 设置对角线元素为零
    K(eye(size(K,1))>0)=0;
    % 处理无穷大的情况
    K(find(isinf(K)==1))=0; 
    
    % 获取奇异值矩阵中接近 1 的索引
    ind_s1 = find(abs(diagS-1)<1e-10);
    if isempty(ind_s1) == 0
        K(ind_s1,ind_s1) = 0; % 将接近 1 的位置设置为零
    end

    ind_s0 = find(diagS<1e-10); % 获取奇异值矩阵中小于阈值的索引
    if isempty(ind_s0) == 0
        K(ind_s0,ind_s0) = 0; % 将小于阈值的位置设置为零
    end
    
end
if all(diagS==1)
    dzdx = zeros(D,D); % 如果奇异值全为 1，则返回零矩阵
else
    dzdx = U*symmetric(K'.*(U'*dLdU))*U'; % 计算梯度 dzdx
end
