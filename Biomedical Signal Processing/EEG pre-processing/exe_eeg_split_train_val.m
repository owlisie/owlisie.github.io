%{
Project Created 1, Sep, 2022
Author : 장영인
Sampling EEG Signal
Split Train & Validation Range

Date of last Update : 1, Sep, 2022
Update list :
    - v 1.0 : 1, Sep, 2022
    Project created
%}

clc
clearvars
close all

cd 'C:\Users\kbri\Desktop\Q\Project\Python\dataset\EEG Epilepsy Datasets\'
clc

fileName = 'EEG_Epliepsy.csv';
trainfileName = 'EEG_Epliepsy_train.csv';
valfileName = 'EEG_Epliepsy_val.csv';

eeg_data = [];
train_eeg = [];
val_eeg = [];

hz = 200;
sec = 1;
moving = 0.5;
iter = (5.12-sec)/moving;

folderList = ["ictal", "interictal", "preictal"];
label = [0, 1, 2];

for i = 1:length(folderList)
    fol = folderList(i);
    fileinfo = dir(strcat('C:\Users\kbri\Desktop\Q\Project\Python\dataset\EEG Epilepsy Datasets\', fol)); 
    for j = 1:length(fileinfo)
        file = fileinfo(j).name;
        if contains(file, '.mat')
            dataFile = strcat('C:\Users\kbri\Desktop\Q\Project\Python\dataset\EEG Epilepsy Datasets\', fol, '\', file);
            dataMat = load(dataFile);
            dataSig = struct2cell(dataMat);
            for k = 0:iter
                eeg_split = dataSig{1, 1}([k*moving*hz+1:k*moving*hz + sec*hz]);
                eeg_split = [eeg_split; i];
                eeg_data = [eeg_data; eeg_split'];
            end
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



