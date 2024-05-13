function res = vl_myforbackward(net, x, dzdy, res, varargin)
% VL_SIMPLENN  Evaluates a simple GrNet
% VL_MYFORBACKWARD 评估一个简单的GrNet网络
% 这个函数通过执行前向和反向传播来评估网络
% 输入:
% - net: 网络结构体，包含网络层的详细信息
% - x: 输入数据
% - dzdy: 反向传播的导数，对于测试模式可省略
% - res: 存储中间结果的结构体
% - varargin: 可选的输入参数

opts.res = [] ;
opts.conserveMemory = false ; % 是否保存内存
opts.sync = false ; % 是否同步GPU操作
opts.disableDropout = false ; % 是否禁用dropout
opts.freezeDropout = false ;  % 是否冻结dropout
opts.accumulate = false ; % 是否累积梯度
opts.cudnn = true ; % 是否使用CuDNN库
opts.skipForward = false; % 是否跳过前向传播
opts.backPropDepth = +inf ; % 反向传播的深度
opts.epsilon = 5e-5; % 数值稳定性参数
opts.p = 10; % 特定于'orthmap'层的参数

% opts = vl_argparse(opts, varargin);

n = numel(net.layers) ; % 网络层数

if (nargin <= 2) || isempty(dzdy)
  doder = false ; % 如果没有提供dzdy，不执行反向传播
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ; % 检查是否在GPU模式

if nargin <= 3 || isempty(res)  % 如果没有提供res，初始化它
  res = struct(...
    'x', cell(1,n+1), ... %表示每一层的输出，即前向传播后得到的结果。
    'dzdx', cell(1,n+1), ... %对于网络中的每一层，它存储了损失函数对该层输出的梯度值
    'dzdw', cell(1,n+1), ...%表示每一层的参数（权重）相对于损失函数的梯度。在反向传播中，用于更新网络参数。
    'aux', cell(1,n+1), ... %辅助变量，用于存储一些额外的中间结果，可能会在网络的前向或反向传播中用到。
    'time', num2cell(zeros(1,n+1)), ... %记录每一层前向传播的时间，用于性能分析和优化。
    'backwardTime', num2cell(zeros(1,n+1))) ; %记录每一层反向传播的时间，同样用于性能分析和优化。
end
if ~opts.skipForward
  res(1).x = x ;  % 设置输入数据
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------
%前向传播工作
for i=1:n
  if opts.skipForward, break; end;  % 如果需要跳过前向传播，则终止循环
  l = net.layers{i} ;  % 获取当前层的信息
  res(i).time = tic ;  % 记录当前层的前向传播时间

  % 根据当前层的类型执行相应的前向传播操作
  switch l.type
    case 'frmap'
      res(i+1).x = vl_myfrmap(res(i).x, weight) ; 
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, weight) ; 
    case 'reorth'
      [res(i+1).x, res(i)] = vl_myreorth(res(i)) ;      
    case 'pooling'
      [res(i+1).x, res(i)] = vl_mypooling(res(i), l.pool) ;

    case 'orthmap'
      [res(i+1).x, res(i)] = vl_myorthmap(res(i), opts.p) ;   
    case 'projmap'
      res(i+1).x = vl_myprojmap(res(i).x) ;  
      case 'LPP' % LPP层的正向传播已经正确实现
      res(i+1).x = LPP(res(i).x) ;  % 这里已经执行了LPP的正向传播操作，映射输入数据到低维空间
    case 'softmaxloss'
      res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ; 
    case 'custom'  % 自定义层的前向传播
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;  % 如果遇到未知的层类型，则报错
  end
  % optionally forget intermediate results 
  % 可选地忘记中间结果
  forget = opts.conserveMemory ;
  %接下来，检查是否需要执行反向传播（即是否提供了 dzdy 参数），如果没有提供 dzdy 
  % 或者当前层的类型是 "relu"，则保持 forget 为 true，这意味着中间结果会被保留。
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;

  %如果当前层的类型是 "loss" 或 "softmaxloss"，
  % 则将 forget 设置为 false，这意味着中间结果不会被保留。
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  
  %如果当前层没有设置 rememberOutput 属性，或者设置了但为 false，
  % 则将 forget 设置为 true，这也意味着中间结果会被保留。
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
 
  %综合起来，forget 变量确定了是否应该忘记（即不保留）当前层的中间结果。
  %如果 forget 为 true，则中间结果会被清除以节省内存；
  %如果为 false，则中间结果会保留，以便后续使用。%


  %%如果在GPU模式下且需要同步操作，则等待GPU操作完成
  if forget  
    res(i).x = [] ;
  end

  %这应该使事情变慢，但在MATLAB 2014a上，对于任何体面的性能都是必要的。
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end

  res(i).time = toc(res(i).time) ; %% 记录当前层的前向传播时间
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------
%反向传播工作
if doder
  res(n+1).dzdx = dzdy ; % 设置最后一层的导数

  % 从最后一层开始反向传播，直至达到指定的深度  
  for i=n:-1:max(1, n-opts.backPropDepth+1)  
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'frmap'
        [res(i).dzdx, res(i).dzdw] =  vl_myfrmap(res(i).x, weight, res(i+1).dzdx) ;
      case 'fc'
        [res(i).dzdx, res(i).dzdw]  = vl_myfc(res(i).x, weight, res(i+1).dzdx) ; 
      case 'reorth'
        res(i).dzdx = vl_myreorth(res(i), res(i+1).dzdx) ;
      case 'pooling'
        res(i).dzdx = vl_mypooling(res(i), l.pool, res(i+1).dzdx) ;
     
      case 'orthmap'
        res(i).dzdx = vl_myorthmap(res(i), opts.p, res(i+1).dzdx) ; 
      case 'projmap'
        res(i).dzdx = vl_myprojmap(res(i).x, res(i+1).dzdx) ;
      case 'LPP' % LPP层的特殊处理
        % 因为LPP层没有直接的反向传播实现，这里简单地将上游梯度传递给下一层
        res(i).dzdx = res(i+1).dzdx; % 直接传递梯度而不进行LPP层特有的操作
      case 'softmaxloss'
        res(i).dzdx = vl_mysoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'custom' % 自定义层的反向传播
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    
    % 如果需要保存内存，则清空中间结果的梯度
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end

