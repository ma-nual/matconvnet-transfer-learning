function net = cnn_prepareNet(net,opts)

fc8l = (cellfun(@(a) strcmp(a.name, 'layer7'), net.layers)==1);

nCls = 2;
sizeW = size(net.layers{fc8l}.weights{1});

if sizeW(4)~=nCls
  net.layers{fc8l}.weights = {zeros(sizeW(1),sizeW(2),sizeW(3),nCls,'single'), ...
    zeros(1, nCls, 'single')};
end

net.layers{end} = struct('name','loss', 'type','softmaxloss') ;

% convert to dagnn dagnn网络，还需要添加下面这几层才能训练
% net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
% 
% net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
%     {'prediction','label'}, 'top1err') ;
% net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
%     'opts', {'topK',5}), ...
%     {'prediction','label'}, 'top5err') ;
