function [show_b, show_c] = rand_sample(tru,fal,pre_1,pre_2)
tru=reshape(tru,10,10,1253);
fal=reshape(fal,10,10,2347);
% t1=[1 55 2 15 22 3 17 54 23 102 69 4 18 112 243 70 39 5 103 19 113 40 79 114 6 41 99 56 80 7 20 74 8 21 105 9 106 81 107 42 108 94 109 57 82 110 71 115 64 83 116 10 24 95 88 117 11 84 100 72 89 43 118 12 126 131 119 132 120 61 90 73 121 44 62 91 122 45 63 92 75 46 93 65 123 47 66 48 76 49 130 124 77 50 779 125 832 87 51 78];
% t2=[17 1 22 31 56 38 9 2 47 39 97 62 84 57 48 85 32 10 58 3 40 86 11 98 33 69 94 87 41 99 23 49 63 28 100 42 117 4 88 118 35 119 12 89 120 101 43 121 5 24 122 13 123 70 124 44 125 107 25 126 102 71 127 138 45 128 90 14 112 103 50 129 16 130 91 131 108 72 113 132 51 109 64 29 133 104 73 136 110 52 26 59 53 135 74 95 34 137 54 92];
t1=randperm(1253,100);
t2=randperm(2347,100);
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


