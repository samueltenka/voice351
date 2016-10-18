clear all;
clc;
Sound = audioread('plain-speech.wav');

fs = 8000;
X = Sound(:,2);
X(abs(X)<0.03)=0;
%sound(X,fs);
x = length(X);

cursor=1;
while (cursor< (x-3000))
    temp=X(cursor:cursor+2999);
    y=significant_piece(temp);
    if y==1
        sound(temp,fs);
        pause(1.2);
        cursor=cursor+1999;
    else
        cursor=cursor+1;
    end
end