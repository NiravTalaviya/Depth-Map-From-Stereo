%%read image
I = imread('test.jpg');

%show original image
figure();
imshow(I);

%height width count
height=size(I,1);
width=size(I,2);

for i=1:height
    for j=1:width
        left=max(0,j-hs);
        top=max(0,i-hs);
        right=min(width,j+hs);
        bottom=min(height,i+hs);
    end
end