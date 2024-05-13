function net = grnet_init_afew(varargin)
% grnet_init Initialize a grnet
% 设置随机数生成器的种子
rng('default');
rng(0) ;

% 2 Blocks (achives the highest accuracy of 34.50% on AFEW)
opts.datadim = [400, 300, 100, 50]; % 数据维度
opts.skedim = [8, 8, 8, 8]; % 顶层的特征图维度
opts.pool = [2, 2, 2, 2, 2]; % 池化层参数
opts.layernum = length(opts.datadim)-2; %设计了网络中的层数，值为2

Winit = cell(opts.layernum+1,1); % 初始化权重矩阵


% % 2 Blocks
% opts.datadim = [400, 300, 100, 50];
% opts.skedim = [16, 16, 16, 16];
% opts.pool = [2, 2, 2, 2, 2];
% opts.layernum = length(opts.datadim)-2;
% Winit = cell(opts.layernum+1,1);


% 1 Block
% opts.datadim = [400, 100, 50];
% opts.skedim = [16, 16,16];
% opts.pool = [2, 2,2];
% opts.layernum = length(opts.datadim)-1;
% Winit = cell(opts.layernum+1,1);

%遍历每一层，初始化权重
for iw = 1 :  opts.layernum
    for i_s = 1 : opts.skedim(iw) %循环遍历当前层中的每个特征图。
        
        if iw ==1
            A = rand(opts.datadim(iw));
        else
            A = rand(opts.datadim(iw)/2);
        end
         % 使用 SVD 初始化权重
        [U1, S1, V1] = svd(A * A');
        Winit{iw}.w(:,:,i_s) = U1(:,1:opts.datadim(iw+1))';
    end
end

%初始化全连接层权重
f=1/100 ;
classNum = 7;

fdim = opts.datadim(end)*opts.datadim(end)*opts.skedim(end);


theta = f*randn(fdim, classNum, 'single'); %fdim=20000
Winit{iw+1}.w  = theta';
% 创建网络结构
net.layers = {} ;
net.layers{end+1} = struct('type', 'frmap') ;%添加了一个新的层，类型是 'frmap'
net.layers{end}.weight = Winit{1}.w;
%依次添加了几个新的层
% 包括 'reorth'、'projmap'、'pooling' 和 'orthmap'
net.layers{end+1} = struct('type', 'reorth') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'pooling') ;
net.layers{end}.pool = opts.pool(1);
net.layers{end+1} = struct('type', 'orthmap') ;
%添加了另一个 'frmap' 层，并设置权重为 Winit{2}.w
net.layers{end+1} = struct('type', 'frmap') ;
net.layers{end}.weight = Winit{2}.w;

% 添加了 'reorth'、'projmap'、'pooling' 和 'orthmap' 层。
net.layers{end+1} = struct('type', 'reorth') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'pooling') ;
net.layers{end}.pool = opts.pool(2);
net.layers{end+1} = struct('type', 'orthmap') ;
%最后添加了 'projmap'、'fc' 和 'softmaxloss' 层，
% 并设置了最后一层的权重为 Winit{end}.w。
net.layers{end+1} = struct('type', 'LPP');
net.layers{end+1} = struct('type', 'projmap') ;

net.layers{end+1} = struct('type', 'fc');

net.layers{end}.weight = Winit{end}.w;
net.layers{end+1} = struct('type', 'softmaxloss') ;
