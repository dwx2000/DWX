function Y = vl_mysoftmaxloss(X,c,dzdy)
%softmax layer
% vl_mysoftmaxloss: Softmax损失层
%
% 用法：
%   Y = vl_mysoftmaxloss(X, c)
%   Y = vl_mysoftmaxloss(X, c, dzdy)
%
% 输入参数：
%   - X: 输入数据
%   - c: 类别标签
%   - dzdy: 反向传播的导数，对于测试模式可省略
%
% 输出参数：
%   - Y: Softmax损失
%
% 详细解释：
%   vl_mysoftmaxloss 函数实现了Softmax损失层的前向和反向传播操作。对于前向传播，输入数据 X 
%   经过Softmax运算后，计算Softmax损失；对于反向传播，根据类别标签 c 和反向传播的导数 dzdy，
%   计算Softmax损失层的梯度。


% class c = 0 skips a spatial location
mass = single(c > 0) ; % 计算非零类别的位置
mass = mass';

% convert to indexes
c_ = c - 1 ; % 将类别标签转换为索引
for ic = 1  : length(c)
    c_(ic) = c(ic)+(ic-1)*size(X,1); % 计算每个类别在X中的索引
end

% compute softmaxloss
Xmax = max(X,[],1) ;  % 计算每列的最大值
ex = exp(bsxfun(@minus, X, Xmax)) ; % 计算指数部分

% s = bsxfun(@minus, X, Xmax);
% ex = exp(s) ;
% y = ex./repmat(sum(ex,1),[size(X,1) 1]);

%n = sz(1)*sz(2) ;
if nargin < 3
  t = Xmax + log(sum(ex,1)) - reshape(X(c_), [1 size(X,2)]) ; % 计算Softmax损失
  Y = sum(sum(mass .* t,1)) ; % 计算总损失
else
  Y = bsxfun(@rdivide, ex, sum(ex,1)) ; % 计算Softmax输出
  Y(c_) = Y(c_) - 1; % 计算梯度
  Y = bsxfun(@times, Y, bsxfun(@times, mass, dzdy)) ; % 计算梯度乘积
end