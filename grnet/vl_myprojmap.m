function Y = vl_myprojmap(X, dzdy)
%projection mapping(ProjMap) layer
% vl_myprojmap: ProjMap（投影映射）层
%
% 用法：
%   Y = vl_myprojmap(X, dzdy)
%
% 输入参数：
%   - X: 输入数据
%   - dzdy: 反向传播的导数，对于测试模式可省略
%
% 输出参数：
%   - Y: 输出数据，经过投影映射处理后的结果
%
% 详细解释：
%   vl_myprojmap 函数对输入数据进行投影映射处理，计算输出数据 Y。如果输入参数
%   dzdy 为空，则将输入数据 X 进行投影映射并返回结果 Y；如果输入参数 dzdy 不
%   为空，则根据 dzdy 对 X 进行反向传播，并计算输出数据 Y


[n1,n2,n3,n4] = size(X); % 获取输入数据的维度

Y = zeros(n1,n1,n3,n4); % 初始化输出 Y

if nargin < 2
    % 如果 dzdy 为空，则进行前向传播计算
    for ix = 1: n3
        if n4 == 1
            % 对每个矩阵进行投影映射处理
            Y(:,:,ix) = X(:,:,ix)*X(:,:,ix)';
        else
            for iy = 1 : n4
                Y(:,:,ix,iy) = X(:,:,ix,iy)*X(:,:,ix,iy)';
            end
        end
    end
else
    % 如果 dzdy 不为空，则进行反向传播计算
    Y = zeros(n1,n2,n3,n4);  % 初始化输出 Y
    [n5,n6,n7,n8] = size(dzdy);
    
    
    for ix = 1: n3
        d_t = dzdy(:,ix);     % 获取当前矩阵的导数
       
        
        if n7 ==1
            % 如果导数为向量，则将其转换为矩阵
            if n4 ==1
                d_t = reshape(d_t,[n1 n1]);
                Y(:,:,ix) = 2*d_t*X(:,:,ix); % 计算梯度并更新 Y
            else
                d_t = reshape(d_t,[n1 n1 n4]);
               
                for iy = 1 : n4
                    Y(:,:,ix,iy) = 2*d_t(:,:,iy)*X(:,:,ix,iy); % 计算梯度并更新 Y
                end
            end
        else
            %如果导数为矩阵，则直接计算梯度并更新
            for iy = 1 : n4
                Y(:,:,ix,iy) = 2*dzdy(:,:,ix,iy)*X(:,:,ix,iy);
            end
        end

    end
end
