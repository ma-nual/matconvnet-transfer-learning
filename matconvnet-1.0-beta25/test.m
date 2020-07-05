net = load('imagenet-vgg-f.mat');

%读入并预处理一幅图像
im = imread('peppers.png'); %读入matlab自带图像
im_ = imresize(single(im), net.meta.normalization.imageSize(1:2)); %转换图像的数据类型，规范化输入图像的宽高
im_ = im_ - net.meta.normalization.averageImage; %将输入图像减去模型均值

%运行CNN
res = vl_simplenn(net, im_); %返回一个res结构的输出网络层

%展示分类结果
scores = squeeze(gather(res(end).x)); %得到图像属于各个分类的概率
[bestScore, best] = max(scores); %得到最大概率值，及其索引值
figure(1) ; clf ; imagesc(im); %显示图像
title(sprintf('%s (%d), score %.3f',net.meta.classes.description{best}, best, bestScore)); %添加标题——"种类(第几类),所属该类的概率"