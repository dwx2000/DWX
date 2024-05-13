function [Y, Y_w] = vl_myfc(X, W, dzdy)
%fully-connected convolutional (FC) layer
% vl_myfc: 全连接卷积（FC）层
%
% 用法：
%   [Y, Y_w] = vl_myfc(X, W, dzdy)
%
% 输入参数：
%   - X: 输入数据
%   - W: 权重参数
%   - dzdy: 反向传播的导数，对于测试模式可省略
%
% 输出参数：
%   - Y: 输出数据
%   - Y_w: 权重参数的梯度
%
% 详细解释：
%   vl_myfc 函数实现了全连接卷积（FC）层的前向和反向传播操作。对于前向传播，输入数据 X
%   经过权重参数 W 的线性变换得到输出数据 Y；对于反向传播，根据输入数据 X 和反向传播的
%   导数 dzdy，计算权重参数 W 的梯度 Y_w。


[n1,n2,n3,n4] = size(X); % 获取输入数据的维度

X_t = zeros(n1*n2*n4,n3); % 初始化转置矩阵

for ix = 1 : n3
    x_t = X(:,:,ix,:); % 获取当前数据
    X_t(:,ix) = x_t(:); % 将当前数据转换为列向量并存储在转置矩阵中
end
if nargin < 3
    Y = W * X_t; % 计算前向传播结果
else
    Y = W' * dzdy; % 计算反向传播结果
    Y_w = dzdy*X_t'; % 计算权重参数的梯度
end