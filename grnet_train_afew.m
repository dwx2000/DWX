function [net, info] = grnet_train_afew(net, gr_train, opts)
% GRNET_TRAIN_AFEW 使用指定的训练数据和选项训练 GRNet 模型。
%
% 输入：
%   - net：初始的 GRNet 模型。
%   - gr_train：包含训练数据的结构体。
%   - opts：训练选项。
%
% 输出：
%   - net：训练后的 GRNet 模型。
%   - info：训练过程的信息

% 初始化选项中的错误标签
opts.errorLabels = {'top1e'};

% 设置训练和验证数据索引
opts.train = find(gr_train.gr.set==1) ;
opts.val = find(gr_train.gr.set==2) ;

for epoch=1:opts.numEpochs
    learningRate = opts.learningRate(epoch);  % 获取当前周期的学习率
    
    % fast-forward to last checkpoint  从上一个检查点继续训练或加载模型
    modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
    modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
    if opts.continue
        if exist(modelPath(epoch),'file')
            if epoch == opts.numEpochs
                load(modelPath(epoch), 'net', 'info') ; % 如果存在当前周期的模型文件，且为最后一个周期，则加载该模型和训练信息
            end
            continue ;
        end
        if epoch > 1
            fprintf('resuming by loading epoch %d\n', epoch-1) ;
            load(modelPath(epoch-1), 'net', 'info') ;
        end
    end
    % 打乱训练数据索引
    train = opts.train(randperm(length(opts.train))) ; % shuffle
    val = opts.val;
    
    %process_epoch这个函数负责处理单个训练或验证周期，执行前向传播、后向传播，并更新模型权重。
    [net,stats.train] = process_epoch(opts, epoch, gr_train, train, learningRate, net) ;
     % 处理训练周期
    [net,stats.val] = process_epoch(opts, epoch, gr_train, val, 0, net) ;
    % 处理验证周期，学习率设置为0表示不进行权重更新
    
    
    % 设置评估模式标志
    evaluateMode = 0;
    % 根据评估模式选择数据集
    if evaluateMode, sets = {'train'} ; else sets = {'train', 'val'} ; end
    
    for f = sets
        f = char(f) ;
        n = numel(eval(f)) ; %
        info.(f).objective(epoch) = stats.(f)(2) / n ; % 计算目标函数值
        info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
    end
    % 如果不是评估模式，则保存模型和训练信息
    if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end

    % 创建新图形，清除当前图形
    figure(1) ; clf ;
    % 设置是否有错误标志
    hasError = 1 ;
    % 创建子图
    subplot(1,1+hasError,1) ;
    if ~evaluateMode
        % 绘制训练目标函数值
        semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
        hold on ;
    end
    semilogy(1:epoch, info.val.objective, '.--') ; % 绘制验证目标函数值
    xlabel('training epoch') ; ylabel('energy') ;

    grid on ; % 开启网格
    h=legend(sets) ;  % 显示图例
    set(h,'color','none');  % 设置图例背景透明
    title('objective') ;
    if hasError
        subplot(1,2,2) ; leg = {} ;  % 如果存在错误，创建第二个子图
        if ~evaluateMode
            % 绘制训练错误率
            plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
            hold on ; % 保持图像，以便添加下一个绘图
            leg = horzcat(leg, strcat('train ', opts.errorLabels)) ; % 添加图例标签
        end
        plot(1:epoch, info.val.error', '.--') ;  % 绘制验证错误率
        leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
        set(legend(leg{:}),'color','none') ;
        grid on ;
        xlabel('training epoch') ; ylabel('error') ;
        title('error') ;
    end
    drawnow ; drawnow ; % 立即绘制图形
%     if epoch == 100
    print(1, modelFigPath, '-dpdf') ;  % 将图形保存为PDF文件
%     end
end



function [net,stats] = process_epoch(opts, epoch, gr_train, trainInd, learningRate, net)
% PROCESS_EPOCH 处理单个训练或验证周期。
%
% 输入：
%   - opts：训练选项。
%   - epoch：当前周期编号。
%   - gr_train：包含训练数据的结构体。
%   - trainInd：当前周期的训练样本索引。
%   - learningRate：当前周期的学习率。
%   - net：GRNet 模型。
%
% 输出：
%   - net：更新后的 GRNet 模型。
%   - stats：周期统计信息。

% 确定是训练还是验证
training = learningRate > 0 ;
if training, mode = 'training' ; else mode = 'validation' ; end

% 初始化统计信息
stats = [0 ; 0 ; 0] ;
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;% 如果存在 GPU，则将值转换为 GPU 数组
else
    one = single(1) ; % 否则使用普通的单精度数组
end

batchSize = opts.batchSize; % 获取批量大小
errors = 0; % 初始化错误数量
numDone = 0 ; % 初始化完成的样本数量
% 获取第一个训练样本的路径
grPath = [gr_train.grDir '\' gr_train.gr.name{trainInd(1)}];
load(grPath); [n1,n2] = size(Y1);  % 加载第一个训练样本的数据，并获取其大小 Y1是加载的数据矩阵。

% 处理每个 batch 的数据

for ib = 1 : batchSize : length(trainInd)
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ; % 开始计时
    res = []; % 初始化结果
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;
    else
        batchSize_r = batchSize;
    end
    gr_data = zeros(n1,n2,batchSize_r);  % 初始化当前 batch 的数据矩阵

    gr_label = zeros(batchSize_r,1);  % 初始化当前 batch 的标签向量
    for ib_r = 1 : batchSize_r
        grPath = [gr_train.grDir '\' gr_train.gr.name{trainInd(ib+ib_r-1)}];
        load(grPath); gr_data(:,:,ib_r) = Y1; % 加载当前训练样本的数据

        gr_label(ib_r) = gr_train.gr.label(trainInd(ib+ib_r-1)); % 获取当前训练样本的标签
        
    end
    %这里设置了网络最后一层的class属性，以便计算损失函数时使用真实标签。
    net.layers{end}.class = gr_label ;
    
    %forward/backward grnet
    %one = single(1) ; % 否则使用普通的单精度数组
    % 如果是训练模式，设置dzdy为1，否则为空；这将决定是否执行反向传播
    if training, dzdy = one; else dzdy = [] ; end

    % 执行GRNet的前向和反向传播
    res = vl_myforbackward(net, gr_data, dzdy, res) ;
    
    %accumulating graidents
    if numGpus <= 1
        % 对于单GPU或CPU，直接在当前设备上累积梯度并更新权重
        [net,res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
        if isempty(mmap)
            mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
        end
        write_gradients(mmap, net, res) ;
        labBarrier() ;
        [net,res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end
    
    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', gr_label)) ;
    
    numDone = numDone + batchSize_r ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    stats = stats+[batchTime ; res(end).x ; error]; % works even when stats=[]
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    
    fprintf(' error: %.5f', stats(3)/numDone) ;
    fprintf(' obj: %.5f', stats(2)/numDone) ;
    
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf('\n') ;
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=numel(net.layers):-1:1
    if isempty(res(l).dzdw)==0
        if ~isfield(net.layers{l}, 'learningRate')
            net.layers{l}.learningRate = 1 ;
        end
        if ~isfield(net.layers{l}, 'weightDecay')
            net.layers{l}.weightDecay = 1;
        end
        thisLR = lr * net.layers{l}.learningRate ;
        
        if isfield(net.layers{l}, 'weight')
            if strcmp(net.layers{l}.type,'orthmap')==1 ...
                    || strcmp(net.layers{l}.type,'reorthmap')==1 
                W=net.layers{l}.weight;
                Wgrad = (1/batchSize)*res(l).dzdw;

                
                for iw = 1 : size(W,3)
                    W1 = W(:,:,iw);
                    W1grad = Wgrad(:,:,iw);
                    
                    %gradient update on PSD manifolds
                    problemW1.M = symfixedrankYYfactory(size(W1,1), size(W1,2));
                    
                    W1Rgrad = (problemW1.M.egrad2rgrad(W1', W1grad'))';
                    net.layers{l}.weight(:,:,iw) = (problemW1.M.retr(W1', -thisLR*W1Rgrad'))'; %%!!!NOTE
                    
                end
            else
                net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize)* res(l).dzdw ;
            end
        end
    end
end



