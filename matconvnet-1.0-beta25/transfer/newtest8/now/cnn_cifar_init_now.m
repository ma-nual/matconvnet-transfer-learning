function net = cnn_cifar_init_now(varargin)
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

lr = [.1 2] ;

net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(3,3,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(3,3,10,20, 'single'), zeros(1,20,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ; % Emulate caffe

% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,20,2, 'single'), zeros(1,2,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;

% % Block 4
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{0.05*randn(1,1,20,2, 'single'), zeros(1,2,'single')}}, ...
%                            'learningRate', lr, ...
%                            'stride', 1, ...
%                            'pad', 0) ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

net.meta.inputSize = [10 10 1] ;
net.meta.trainOpts.learningRate = [0.05*ones(1,60) 0.005*ones(1,60) 0.0005*ones(1,80)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 20 ;
% net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;
net.meta.trainOpts.numEpochs = 200 ;

net = vl_simplenn_tidy(net) ;

switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end