%% 测试所有图片
run('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;
% load('E:\store\artificial\matconvnet-1.0-beta25\net-epoch-300.mat') ;
load('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now\net-epoch-19.mat') ;
mycifar_batch1_2_imdb = load('E:\store\artificial\matconvnet-1.0-beta25\matconvnet-1.0-beta25\transfer\newtest8\now\imdb.mat');

test_index = find(mycifar_batch1_2_imdb.images.set==3);
test_data = mycifar_batch1_2_imdb.images.data(:,:,:,test_index);
test_label =mycifar_batch1_2_imdb.images.label(test_index);

% test_index1 = find(mycifar_batch1_2_imdb.images.set==4);
% test_data1 = mycifar_batch1_2_imdb.images.data(:,:,:,test_index1);
% test_label1 =mycifar_batch1_2_imdb.images.label(test_index1);

test_data1 = mycifar_batch1_2_imdb.images.vis;

im1=imread('b2.jpg');
% image1=medfilt2(im1,[4 4]);
% image1=im1;
image1=medfilt2(im1);

% B1=im2bw(image1);
% se=strel('square',20);
% C1=imdilate(B1,se);
% D1=imerode(C1,se);
% D1=1-D1;
% D1=im2double(D1);
% A21=im2double(image1);
% image1=times(D1,A21);

net = vl_simplenn_tidy(net) ;
net.layers{1,end}.type = 'softmax';

% i=imread('ASA_IMP_2447_zhanglixiang.jpg');
% imshow(i);

for i = 1:length(test_label)
    i
    im_ = test_data(:,:,:,i);
%     im_ = im_ - mycifar_batch1_2_imdb.images.data_mean;
%     res = vl_simplenn(net, im_,[], [], ...
%                       'accumulate', 0, ...
%                       'mode', 'test', ...
%                       'backPropDepth', Inf, ...
%                       'sync', 0, ...
%                       'cudnn', 1);
    res = vl_simplenn(net,im_);
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores) ;
    pre(i) = best;
end

test_vis = image1;
% m=1;
% n=1;
% for d = 1:10:580
%     for t = 1:10:580
%         im=imcrop(test_data1,[t,d,19,19]);
%         p(m:m+19,n:n+19)=im;
%         n=n+20;
%     end
%     m=m+20;
%     n=1;
% end
% p = reshape(p,20,20,1,3364);

n=1;
for d = 1:10:591
    for t = 1:10:591
        im=imcrop(test_data1,[t,d,9,9]);
        p(1:10,1:10,1,n)=im;
        n=n+1;
    end
end
p=single(p);
for h = 1:(n-1)
    h
    im_1 = p(:,:,:,h);
%     im_ = im_ - mycifar_batch1_2_imdb.images.data_mean;
%     res1 = vl_simplenn(net, im_1,[], [], ...
%                       'accumulate', 0, ...
%                       'mode', 'test', ...
%                       'backPropDepth', Inf, ...
%                       'sync', 0, ...
%                       'cudnn', 1);
    res1 = vl_simplenn(net,im_1);
    scores1 = squeeze(gather(res1(end).x)) ;
    [bestScore1, best1] = max(scores1) ;
    pre1(h) = best1;
end

% for t = 1:length(test_label1)
%     t
%     im_1 = test_data1(:,:,:,t);
% %     im_ = im_ - mycifar_batch1_2_imdb.images.data_mean;
% %     res1 = vl_simplenn(net, im_1,[], [], ...
% %                       'accumulate', 0, ...
% %                       'mode', 'test', ...
% %                       'backPropDepth', Inf, ...
% %                       'sync', 0, ...
% %                       'cudnn', 1);
%     res1 = vl_simplenn(net,im_1);
%     scores1 = squeeze(gather(res1(end).x)) ;
%     [bestScore1, best1] = max(scores1) ;
%     pre1(t) = best1;
% end
% 
% pp1=zeros(30,30);
% s=1;
% for m=1:30
%     for n=1:30
%         pp1(m,n)=pre1(1,s);
%         s=s+1;   
%     end
% end
% pp1=pp1';
% % pp1=medfilt2(pp1);
%  for p=1:20:600
%      for q=1:20:600
% % 除以30将截取的待测图像位置与分类结果位置联系起来
%          h=(p-1)/20+1;l=(q-1)/20+1;
%          if (pp1(h,l))==2
%              image1(q:q+19,p:p+19)=0;
%              ia1=image1;
% %             ia2=im2double(ia1);
% %              ia=ia2+E2;
%              figure(3);
%              imshow(ia1);
% %              title('RobustBoost');
%          end
%     end
%  end
 
pp1=zeros(60,60);
s=1;
for x=1:60
    for y=1:60
        pp1(x,y)=pre1(1,s);
        s=s+1;   
    end
end
pp1=pp1';
% pp1=medfilt2(pp1);
 for v=1:10:591
     for q=1:10:591
% 除以30将截取的待测图像位置与分类结果位置联系起来
         j=(v-1)/10+1;l=(q-1)/10+1;
         if (pp1(j,l))==2
             image1(q:q+9,v:v+9)=test_vis(q:q+9,v:v+9);
             ia1=image1;
             figure(3);
             imshow(ia1);
         else if (pp1(j,l))==1
                 image1(q:q+9,v:v+9)=0;
                 ia1=image1;
                 figure(3);
                 imshow(ia1);
%              title('RobustBoost');
             end
         end
    end
 end

% A11=image1;
% A21=A11;
% A41=A11;
% [m,n]=size(A11);
% for i=1:m                          %像素判断，像素小于60置0，否则置255
%     for j=1:n
%         if ((A11(i,j)>60))
%            A11(i,j)=0;
%         else
%             A11(i,j)=255;
%         end
%     end
% end
% B1=im2bw(A11);               %把A11灰度图像转化为二值图像
% se=strel('square',38);
% C1=imdilate(B1,se);          %二值图像膨胀
% D1=imerode(C1,se);           %二值图像腐蚀
% D1=1-D1;                     %像素置换
% D1=im2double(D1);            %转换成双精度浮点类型
% A21=im2double(A21);
% E1=times(D1,A21);            %两图像像素分别相乘
% 
% B2=im2bw(A41);               %把A11灰度图像转化为二值图像
% se=strel('square',20);
% C2=imdilate(B2,se);          %二值图像膨胀
% D2=imerode(C2,se);           %二值图像腐蚀
% D2=1-D2;                     %像素置换
% D2=im2double(D2);            %转换成双精度浮点类型 
% A41=im2double(A41);          
% E2=times(D2,A41);            %两图像像素分别相乘
% image1=E1+E2;
% figure(5);
% imshow(image1);
 
% 计算准确率
accurcy = length(find(pre==test_label))/length(test_label);
disp(['accurcy = ',num2str(accurcy*100),'%']);
% figure(7);
% cm=confusionchart(pre,test_label, ...
%     'ColumnSummary','column-normalized', ...
%     'RowSummary','row-normalized');
for t=1:3600
    if(pre(1,t)==2)
        pre_d1{1,t}='农田';
    else if(pre(1,t)==1)
            pre_d1{1,t}='非农田';
        end
    end
end
for t=1:3600
    if(test_label(1,t)==2)
        test_label1{1,t}='农田';
    else if(test_label(1,t)==1)
            test_label1{1,t}='非农田';
        end
    end
end
figure(7);
% cm=confusionchart(pre_d1,test_label1, ...
%     'ColumnSummary','column-normalized');
% cm=confusionchart(test_label1,pre_d1, ...
%     'RowSummary','row-normalized');
cm=confusionchart(test_label1,pre_d1, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
