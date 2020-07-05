function [net, info] = cnn_train_now(varargin)

run('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;

opts.modelType = 'cifar' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = 'E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = 'E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

switch opts.modelType
  case 'mnist'
    net = cnn_mnist_init_now('networkType', opts.networkType) ;
  case 'cifar'
    net = cnn_cifar_init_now('networkType', opts.networkType) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = get_Imdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb,  getBatch(opts),...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.label(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = get_Imdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted

opts.dataDir = 'E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now';

load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now\train.mat')
load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now\val.mat')
load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now\test.mat')
load ('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now\vis.mat')

x1 = train_x;  
x2 = val_x;  
x3 = test_x;
y1 = train_y;  
y2 = val_y;
y3 = test_y;
z = vis_x;

imdb.images.data=[];%ÕºœÒ ˝æ›  
imdb.images.label=[];%ÕºœÒ±Í«©  
imdb.images.set = [] ;%ÕºœÒ…Ë÷√  
% imdb.meta.sets = {'train', 'val', 'test'} ;  
imdb.imageDir.train = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now') ;
imdb.imageDir.dev = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now') ;
imdb.imageDir.test = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now') ;
imdb.imageDir.vis = fullfile('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now') ;

% set = [ones(1,numel(y1)) 2*ones(1,numel(y2)) 3*ones(1,numel(y3)) 4*ones(1,numel(z)/400)];
% data = [x1,x2,x3,z];
% label = [y1,y2,y3,0*ones(1,numel(z)/400)];

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