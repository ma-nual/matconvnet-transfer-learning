function [show_b, show_c] = rand_sample(tru,fal,pre_1,pre_2)
tru=reshape(tru,10,10,1312);
fal=reshape(fal,10,10,2288);
% t1=(1:100);
% t2=(1:100);
t1=randperm(1312,100);
t2=randperm(2288,100);
tru_1=zeros(10,10,100);
fal_1=zeros(10,10,100);
show_b=zeros(1,100);
show_c=zeros(1,100);
for i=1:100
    tru_1(:,:,i)=tru(:,:,t1(i));
    fal_1(:,:,i)=fal(:,:,t2(i));
    show_b(i)=pre_1(t1(i));
    show_c(i)=pre_2(t2(i));
end
tru_1=reshape(tru_1,10,1000);
tru_1=[tru_1(:,1:100);tru_1(:,101:200);tru_1(:,201:300);tru_1(:,301:400);tru_1(:,401:500);...
    tru_1(:,501:600);tru_1(:,601:700);tru_1(:,701:800);tru_1(:,801:900);tru_1(:,901:1000)];
figure(8);
imshow(tru_1);
fal_1=reshape(fal_1,10,1000);
fal_1=[fal_1(:,1:100);fal_1(:,101:200);fal_1(:,201:300);fal_1(:,301:400);fal_1(:,401:500);...
    fal_1(:,501:600);fal_1(:,601:700);fal_1(:,701:800);fal_1(:,801:900);fal_1(:,901:1000)];
figure(9);
imshow(fal_1);

