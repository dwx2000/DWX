function [Y, R] = vl_mypooling(R, pool, dzdy)
%projection pooling (ProjPooling) layers
% vl_mypooling: ProjPooling（投影池化）层
%
% 用法：
%   [Y, R] = vl_mypooling(R, pool, dzdy)
%
% 输入参数：
%   - R: 存储中间结果的结构体
%   - pool: 池化大小
%   - dzdy: 反向传播的导数，对于测试模式可省略
%
% 输出参数：
%   - Y: 输出数据，经过池化处理后的结果
%   - R: 存储中间结果的结构体，更新了池化位置信息
%
% 详细解释：
%   vl_mypooling 函数对输入数据进行投影池化处理，计算输出数据 Y。如果输入参数
%   dzdy 为空，则对输入数据 R.x 进行池化操作并返回结果 Y；如果输入参数 dzdy 不
%   为空，则根据 dzdy 对 R.aux 进行更新，并返回 Y。


X = R.x;
A = R.aux;  % 获取存储中间结果的结构体

[n1,n2,n3,n4] = size(X); % 获取输入数据的维度
Y = zeros(n1/pool,n2/pool,n3,n4);   % 初始化输出 Y
IY = zeros(n1,n2,n3,n4); % 初始化池化位置信息
tI = zeros(pool,pool,n3,n4); % 创建池化位置信息矩阵
tI(1,1,:,:) = 1; % 设置池化位置信息

if isempty(A)==1
     % 如果存储中间结果为空，则进行前向传播计算
    for ix = 1 : pool : n1
        for iy = 1 : pool : n2
            % 提取池化区域并计算平均值
            r_tt = X(ix:ix+(pool-1),iy:iy+(pool-1),:, :);            
            r_tt = reshape(r_tt,[pool*pool n3 n4]);             
            r_mm = mean(r_tt,1);
           
            % 更新 Y
            Y(floor(ix/pool)+1,floor(iy/pool)+1,:,:) =  r_mm;
            
            % 更新池化位置信息
            IY(ix:ix+(pool-1),iy:iy+(pool-1),:,:) = tI;
            
        end
    end
    R.aux = IY;  % 更新存储中间结果的结构体

else
    % 如果存储中间结果不为空，则进行反向传播计算
    Y = zeros(n1,n2,n3,n4);  % 初始化输出 Y
    Y(logical(A)) = dzdy/(pool^2); % 根据导数对存储中间结果进行更新
end

