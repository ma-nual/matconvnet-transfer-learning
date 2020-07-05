function [net, info] = cnn_imagenet_dicnn(varargin)

run('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;

opts.dataDir = 'E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later' ;
opts.expDir  = 'E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later' ;
opts.modelPath = fullfile('E:','store','artificial','matconvnet-1.0-beta25','matconvnet-1.0-beta25','transfer','newtest8','original','net-epoch-200.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

% opts.numFetchThreads = 12 ;

% opts.lite = false ; 
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.modelType = 'cifar' ;
% opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts.train.gpus = []; %填的是GPU索引号，一般不是0就是1
% opts.train.batchSize = 30 ; 
% opts.train.numSubBatches = 4 ;
% opts.train.learningRate = 1e-4 * [ones(1,10), 0.1*ones(1,5)]; 

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

net = load(opts.modelPath);

net.layers=net.net.layers;

net = cnn_prepareNet(net,opts);

if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
%   imdb = get_Imdb('dataDir', opts.dataDir, 'lite', opts.lite) ;
  imdb = get_Imdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes ;
% net.meta.classes.description = imdb.meta.classes ;

% net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

% imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ; %
% if exist(imageStatsPath)
%   load(imageStatsPath, 'averageImage') ;
% else
%     averageImage = getImageStats(opts, net.meta, imdb) ;
%     save(imageStatsPath, 'averageImage') ;
% end

% net.meta.normalization.averageImage = averageImage;

opts.train.train = find(imdb.images.set==1) ;
opts.train.val = find(imdb.images.set==2) ;

net.meta.inputSize = [10 10 1] ;
net.meta.trainOpts.learningRate = [0.05*ones(1,60) 0.005*ones(1,60) 0.0005*ones(1,80)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 20;
net.meta.trainOpts.numEpochs = 200 ;

[net, info] = cnn_train(net, imdb, getBatch(opts), ...
                      'expDir', opts.expDir,...
                      net.meta.trainOpts, ...
                      opts.train, ...
                      'val', find(imdb.images.set == 2)) ;

% [net, info] = cnn_train(net, imdb, getBatchFn(opts, net.meta), ...
%                       'expDir', opts.expDir, ...
%                       opts.train) ; 
                  
% net = cnn_imagenet_deploy(net) ; 
% modelPath = fullfile(opts.expDir, 'net-deployed.mat');

% net_ = net.saveobj() ;
% save(modelPath, '-struct', 'net_') ;
% clear net_ ;

function fn = getBatch(opts)
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% function fn = getBatchFn(opts, meta)
% useGpu = numel(opts.train.gpus) > 0 ;
% 
% bopts.numThreads = opts.numFetchThreads ;
% bopts.imageSize = meta.normalization.imageSize ;
% bopts.border = meta.normalization.border ;
% bopts.averageImage = meta.normalization.averageImage ;

function [images, labels] = getSimpleNNBatch(imdb, batch)

images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.label(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;

% function inputs = getDagNNBatch(opts, useGpu, imdb, batch)

% for i = 1:length(batch)
%     if imdb.images.set(batch(i)) == 1 %1为训练索引文件夹
%         images(i) = strcat([imdb.imageDir.train filesep] , imdb.images.name(batch(i)));
%     else
%         images(i) = strcat([imdb.imageDir.test filesep] , imdb.images.name(batch(i)));
%     end
% end;
% isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
% 
% if ~isVal
%   im = getImageBatch(opts, ...
%                      'prefetch', nargout == 0) ;
% else
%   im = getImageBatch(opts, ...
%                      'prefetch', nargout == 0, ...
%                      'transformation', 'none') ;
% end
% 
% if nargout > 0
%   if useGpu
%     im = gpuArray(im) ;
%   end
%   labels = imdb.images.label(batch) ;
%   inputs = {'input', im, 'label', labels} ;
% end

% function averageImage = getImageStats(opts, meta, imdb)
% 
% train = find(imdb.images.set == 1) ;
% batch = 1:length(train);
% fn = getBatchFn(opts, meta) ;
% train = train(1: 100: end);
% avg = {};
% for i = 1:length(train)-2 %防止数据集不是整数倍出错
%     temp = fn(imdb, batch(train(i):train(i)+99)) ;
%     temp = temp{2};
%     avg{end+1} = mean(temp, 4) ;
% end
% 
% averageImage = mean(cat(4,avg{:}),4) ;
% % 将GPU格式的转化为cpu格式的保存起来（如果有用GPU）
% averageImage = gather(averageImage);

function imdb = get_Imdb(opts)

opts.dataDir = 'E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later';

load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later\train.mat')
load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later\val.mat')
load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later\test.mat')
load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later\vis.mat')

x1 = train_x;  
x2 = val_x;  
x3 = test_x;
y1 = train_y;  
y2 = val_y;
y3 = test_y;
z = vis_x;

imdb.images.data=[];%图像数据  
imdb.images.label=[];%图像标签  
imdb.images.set = [] ;%图像设置  
% imdb.meta.sets = {'train', 'val', 'test'} ;  
imdb.imageDir.train = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later') ;
imdb.imageDir.dev = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later') ;
imdb.imageDir.test = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later') ;
imdb.imageDir.vis = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\later') ;

set = [ones(1,numel(y1)) 2*ones(1,numel(y2)) 3*ones(1,numel(y3))];
data = [x1,x2,x3];
label = [y1,y2,y3];

data = double(data)/255;
vis = double(z)/255;
% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean);
imdb.images.set = set ; 
imdb.images.data = single(reshape(data,10,10,1,[]));
% imdb.images.data_mean = dataMean;
imdb.images.label = label;
imdb.images.vis = vis;
imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.classes.name = {'cropland' 'town'};
imdb.meta.classes = {'town' 'cropland'};  