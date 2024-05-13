function [Y,R] = vl_myreorth(R, dzdy)
%re-orthonormalization (ReOrth) layer
% vl_myreorth: ReOrth（重新正交化）层
%
% 用法：
%   [Y,R] = vl_myreorth(R, dzdy)
%
% 输入参数：
%   - R: 存储中间结果的结构体，包含输入数据 X
%   - dzdy: 反向传播的导数，对于测试模式可省略
%
% 输出参数：
%   - Y: 输出数据，经过重新正交化处理后的结果
%   - R: 存储中间结果的结构体，包含输出数据 Y 和辅助参数 aux
%
% 详细解释：
%   vl_myreorth 函数对输入数据进行重新正交化处理，确保输出数据的正交性。
%   如果输入结构体 R 的辅助参数 aux 为空，则对输入数据进行 QR 分解，
%   并将计算得到的 Q 矩阵和 R 矩阵存储在 R 的辅助参数中。如果输入结构体
%   R 的辅助参数 aux 不为空，则根据存储的 Q 矩阵和 R 矩阵计算输出数据 Y。
%
% 详细步骤：
%   1. 判断输入结构体 R 的辅助参数 aux 是否为空。
%   2. 如果 aux 为空，则对输入数据 X 进行 QR 分解，计算得到 Q 矩阵和 R 矩阵，
%      并将其存储在 R 的辅助参数 aux 中。
%   3. 如果 aux 不为空，则根据存储的 Q 矩阵和 R 矩阵，计算输出数据 Y。
%
% 示例：
%   % 对输入数据进行重新正交化处理
%   [Y, R] = vl_myreorth(R, dzdy);
%


% 获取输入数据
X = R.x;
% 获取输入数据的维度
[n1,n2,n3,n4] = size(X);

% 初始化输出 Y
Y = zeros(n1,n2,n3,n4);
% 如果辅助参数 aux 为空，则进行 QR 分解
if isempty(R.aux)==1
    % 初始化存储 Q 矩阵和 R 矩阵的数组
    Qs = zeros(n1,n2,n3,n4);
    Rs = zeros(n2,n2,n3,n4);
    for ix = 1  : n3
        % 对每个矩阵进行 QR 分解
        if n4 == 1
            [Qs(:,:,ix),Rs(:,:,ix)] = qr(X(:,:,ix),0); 
            Y(:,:,ix) = Qs(:,:,ix); % 更新输出 Y 为 Q 矩阵
        else
            for iy = 1 : n4
                [Qs(:,:,ix,iy),Rs(:,:,ix,iy)] = qr(X(:,:,ix,iy),0);

                Y(:,:,ix,iy) = Qs(:,:,ix,iy);
            end
        end
    end

    % 将计算得到的 Q 矩阵和 R 矩阵存储在 R 的辅助参数中
    R.aux{1} = Qs;
    R.aux{2} = Rs;
else
    % 如果辅助参数 aux 不为空，
    % 则根据存储的 Q 矩阵和 R 矩阵，计算输出 Y
    Qs = R.aux{1};
    Rs = R.aux{2};
    for ix = 1  : n3
        if n4 == 1
            Q = Qs(:,:,ix); R = Rs(:,:,ix);
            T = dzdy(:,:,ix);
            dzdx = Compute_Gradient_QR(Q,R,T); % 计算梯度
            Y(:,:,ix) =  dzdx;  % 更新输出 Y
        else
            for iy = 1 : n4
                Q = Qs(:,:,ix,iy); R = Rs(:,:,ix,iy);
                T = dzdy(:,:,ix,iy);
                dzdx = Compute_Gradient_QR(Q,R,T);
                Y(:,:,ix,iy) =  dzdx;
            end            
        end
    end
end

% 计算梯度的函数
function dzdx = Compute_Gradient_QR(Q,R,T)
m = size(Q,1);  % 获取 Q 矩阵的行数
dLdC = double(T); % 将 T 转换为 double 类型
dLdQ = dLdC;

S = eye(m)-Q*Q'; % 计算 S 矩阵
dzdx_t0 = Q'*dLdQ; % 计算中间变量
dzdx_t1 = tril(dzdx_t0,-1); % 计算下三角部分
dzdx_t2 = tril(dzdx_t0',-1);
dzdx = (S'*dLdQ+Q*(dzdx_t1-dzdx_t2))*(inv(R))';  % 计算梯度
