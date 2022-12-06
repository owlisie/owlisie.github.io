%{
Project Created 16, Sep, 2022
Author : 장영인

Analyze EEG signal in frequency domain
Extract frequency band from signal

Date of last Update : 16, Sep, 2022
Update list :
    - v 1.0 : 16, Sep, 2022
    Project created
%}

clc
clearvars
close all

cd 'C:\Users\kbri\Desktop\Q\Project\Python\dataset\EEG Epilepsy\'
clc

fileName = 'EEG_Epliepsy.csv';
trainfileName = 'EEG_Epliepsy_train.csv';
valfileName = 'EEG_Epliepsy_val.csv';

eeg_data = [];
train_eeg = [];
val_eeg = [];

fs = 200;
sec = 1;
moving = 0.5;
iter = (5.12-sec)/moving;

folderList = ["ictal", "interictal", "preictal"];
label = [0, 1, 2];

for i = 1:length(folderList)
    fol = folderList(i);
    fileinfo = dir(strcat('C:\Users\kbri\Desktop\Q\Project\Python\dataset\EEG Epilepsy\', fol)); 
    for j = 1:length(fileinfo)
        file = fileinfo(j).name;
        if contains(file, '.mat')
            dataFile = strcat('C:\Users\kbri\Desktop\Q\Project\Python\dataset\EEG Epilepsy\', fol, '\', file);
            dataMat = load(dataFile);
            dataSig = struct2cell(dataMat);
            dataEEG = dataSig{1, 1};
            N = length(dataEEG);
            
            % EEG LAB file extension required
            %{
            figure; 
            [spectrum freqs] = pop_spectopo(dataMat, 1, [0 512], 'EEG');
            [tmp minind] = min(abs(freqs-9));
            [tmp maxind] = min(abs(freqs-11));
            alphaPower = mean(spectrum(:, minind:maxind),2);
            figure; 
            topoplot(alphaPower, dataMat.chanlocs, 'maplimits', 'maxmin'); 
            cbar;
            %}
            
            % wavelet toolbox required
            %{
            waveletF = 'db8';
            [C,L] = wavedec(dataEEG,8,waveletF);
   
            cD1 = detcoef(C,L,1);
            cD2 = detcoef(C,L,2);
            cD3 = detcoef(C,L,3);
            cD4 = detcoef(C,L,4);
            cD5 = detcoef(C,L,5); %GAMA
            cD6 = detcoef(C,L,6); %BETA
            cD7 = detcoef(C,L,7); %ALPHA
            cD8 = detcoef(C,L,8); %THETA
            cA8 = appcoef(C,L,waveletF,8); %DELTA
            D1 = wrcoef('d',C,L,waveletF, 1);
            D2 = wrcoef('d',C,L,waveletF, 2);
            D3 = wrcoef('d',C,L,waveletF, 3);
            D4 = wrcoef('d',C,L,waveletF, 4);
            D5 = wrcoef('d',C,L,waveletF, 5); %GAMMA
            D6 = wrcoef('d',C,L,waveletF, 6); %BETA
            D7 = wrcoef('d',C,L,waveletF, 7); %ALPHA
            D8 = wrcoef('d',C,L,waveletF, 8); %THETA
            A8 = wrcoef('a',C,L,waveletF, 8); %DELTA

            Gamma = D5;
            figure; subplot(5,1,1); plot(1:1:length(Gamma),Gamma);title('GAMMA');
            
            Beta = D6;
            subplot(5,1,2); plot(1:1:length(Beta), Beta); title('BETA');
            
            
            Alpha = D7;
            subplot(5,1,3); plot(1:1:length(Alpha),Alpha); title('ALPHA'); 
            
            Theta = D8;
            subplot(5,1,4); plot(1:1:length(Theta),Theta);title('THETA');
            D8 = detrend(D8,0);
            
            Delta = A8;
            %figure, plot(0:1/fs:1,Delta);
            subplot(5,1,5);plot(1:1:length(Delta),Delta);title('DELTA');
            %}
        end
    end
end

writematrix(eeg_data, fileName, 'Writemode', 'append');

rand_list = randperm(size(eeg_data, 1));
temp = eeg_data(rand_list, :);

train_eeg = temp(1:size(eeg_data, 1)*0.8, :);
val_eeg = temp(size(eeg_data, 1)*0.8:end, :);

writematrix(train_eeg, trainfileName, 'Writemode', 'append');
writematrix(val_eeg, valfileName, 'Writemode', 'append');



