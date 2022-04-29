%{
Project Created Nov, 2, 2020
Author : 장영인
Denoise Signal in 3 way
Date of last Update : Nov, 2, 2020
Update list :
    - v 1.0 : Nov, 2, 2020
    Project created
%}   

clc
clearvars
close all

N = 1000; % signal frequency

noise_ID = fopen('noise_list.txt', 'r'); % noise added heart(radar) file list
noise_list = textscan(noise_ID, '%s');
fclose(noise_ID);

mw_ID = fopen('mw_125.txt', 'r'); % mother wavelet list
mw_list = textscan(mw_ID, '%s');
fclose(mw_ID);

for i = 1:length(noise_list{1, 1})
    
    noise_file = noise_list{1, 1}{i, 1};
    noise_matrix = readmatrix(noise_file);
    noise_signal = noise_matrix(:, 1);
    
    denFS = wdenoise(noise_signal, 9); % https://kr.mathworks.com/help/wavelet/ref/wdenoise.html
    
    for j = 1:length(mw_list{1, 1})
        
        mw = mw_list{1, 1}{j, 1};
        [denS1, ent_s, energy] = f_re_signal(noise_file, mw); % Lv9 Reconstruct(cD9)
        
        [LoD, HiD, LoR, HiR] = wfilters(mw);

        % soft threshold denoise : Lv3 고정
        [cD, l] = wavedec(noise_signal, 3, LoD, HiD);
        % 각 coefficient 분리
        cA = cD(1:l(1));
        cD3 = cD(1+l(1):1+l(1)+l(2));
        cD2 = cD(1+l(1)+l(2):1+l(1)+l(2)+l(3));
        cD1 = cD(1+l(1)+l(2)+l(3):end);
        
        sigma3 = median(cD3)/(0.6745); % Universal threhsold value : Donoho et al.
        sigma2 = median(cD2)/(0.6745);
        sigma1 = median(cD1)/(0.6745);
        T3 = sigma3*sqrt(2*log10(N));
        T2 = sigma2*sqrt(2*log10(N));
        T1 = sigma1*sqrt(2*log10(N));
        
        ytsoft3 = wthresh(cD3, 's', T3);
        ytsoft2 = wthresh(cD2, 's', T2);
        ytsoft1 = wthresh(cD1, 's', T1);
        cDSoft1 = [cA; ytsoft3; ytsoft2; ytsoft1];
        
        denSoft1 = waverec(cDSoft1, l, LoR, HiR);
        
        % soft threshold denoise : Lv9
        [cD, l] = wavedec(noise_signal, 9, LoD, HiD);
        % 각 coefficient 분리
        cA = cD(1:l(1));
        cD9 = cD(1+l(1):1+l(1)+l(2));
        cD8 = cD(1+l(1)+l(2):1+l(1)+l(2)+l(3));
        cD7 = cD(1+l(1)+l(2)+l(3):1+l(1)+l(2)+l(3)+l(4));
        cD6 = cD(1+l(1)+l(2)+l(3)+l(4):1+l(1)+l(2)+l(3)+l(4)+l(5));
        cD5 = cD(1+l(1)+l(2)+l(3)+l(4)+l(5):1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6));
        cD4 = cD(1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6):1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7));
        cD3 = cD(1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7):1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7)+l(8));
        cD2 = cD(1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7)+l(8):1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7)+l(8)+l(9));
        cD1 = cD(1+l(1)+l(2)+l(3)+l(4)+l(5)+l(6)+l(7)+l(8)+l(9):end);
        
        sigma9 = median(cD9)/(0.6745);
        sigma8 = median(cD8)/(0.6745);
        sigma7 = median(cD7)/(0.6745);
        sigma6 = median(cD6)/(0.6745);
        sigma5 = median(cD5)/(0.6745);
        sigma4 = median(cD4)/(0.6745);
        sigma3 = median(cD3)/(0.6745);
        sigma2 = median(cD2)/(0.6745);
        sigma1 = median(cD1)/(0.6745);
        T9 = sigma9*sqrt(2*log10(N));
        T8 = sigma8*sqrt(2*log10(N));
        T7 = sigma7*sqrt(2*log10(N));
        T6 = sigma6*sqrt(2*log10(N));
        T5 = sigma5*sqrt(2*log10(N));
        T4 = sigma4*sqrt(2*log10(N));
        T3 = sigma3*sqrt(2*log10(N));
        T2 = sigma2*sqrt(2*log10(N));
        T1 = sigma1*sqrt(2*log10(N));
        
        ytsoft9 = wthresh(cD9, 's', T9);
        ytsoft8 = wthresh(cD8, 's', T8);
        ytsoft7 = wthresh(cD7, 's', T7);
        ytsoft6 = wthresh(cD6, 's', T6);
        ytsoft5 = wthresh(cD5, 's', T5);
        ytsoft4 = wthresh(cD4, 's', T4);
        ytsoft3 = wthresh(cD3, 's', T3);
        ytsoft2 = wthresh(cD2, 's', T2);
        ytsoft1 = wthresh(cD1, 's', T1);
        
        cDSoft2 = [cA; ytsoft9; ytsoft8; ytsoft7; ytsoft6; ytsoft5; ytsoft4; ytsoft3; ytsoft2; ytsoft1];
        
        denSoft2 = waverec(cDSoft2, l, LoR, HiR);

        T = erase(noise_file, '_noise.xlsx'); % file title
        T = erase(T, '_180s_30cm_');
        T = strcat(T, mw);
        title = strcat(T, "_deN.xlsx");
        writematrix(denS1, title, 'Sheet', 'cD8_re');
        writematrix(denS2, title, 'Sheet', 'cD9_re');
        writematrix(denFS, title, 'Sheet', 'Lv9_wde');
        writematrix(denSoft1, title, 'Sheet', 'Lv3_soft');
        writematrix(denSoft2, title, 'Sheet', 'Lv9_soft');
        
    end
    
end