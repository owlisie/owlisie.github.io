clear;
close all;
clc;
eeglab;
EEG = pop_loadset('filename', 'eeglab_data.set','filepath','C:\Users\kbri\Desktop\Q\Project\Matlab\eeglab2022.0\sample_data\');
eeglab redraw;

% parameter initialization
data = EEG.data;
nbchan = EEG.nbchan;
chanlocs = EEG.chanlocs;
data([2 6], :)= []; % EOG
chanlocs([2 6]) = [];
nbchan = nbchan - 2;

% eegh : 수행했던 명령 출력

% select channel
x = data(8, :);
y = data(9, :);
%{
%% mscohere parameters
subplot(2, 3, 1)
%cxy = mscohere(x, y);
cxy = mscohere(x, y, hann(8192), 4096); % window length, overlap size (50%)
plot(cxy)
hold on;
subplot(2, 3, 2)
cxy = mscohere(x, y, hann(4096), 2048); % window shape, (window length), overlab size
plot(cxy)
hold on;
subplot(2, 3, 3)
cxy = mscohere(x, y, hann(2048), 1024);
plot(cxy)
hold on;
subplot(2, 3, 4)
cxy = mscohere(x, y, hann(4096), 3072);
plot(cxy)
hold on;
subplot(2, 3, 5)
cxy = mscohere(x, y, hann(4096), 1024);
plot(cxy)
hold on;
subplot(2, 3, 6)
cxy = mscohere(x, y, hann(4096), 0);
plot(cxy)
hold on;
%}
%%{
subplot(2, 3, 1)
%cxy = mscohere(x, y);
cxy = mscohere(x, y, hann(1024), 512); % window length, overlap size (50%)
plot(cxy)
hold on;
subplot(2, 3, 2)
cxy = mscohere(x, y, hann(2048), 1024); % window shape, (window length), overlab size
plot(cxy)
hold on;
subplot(2, 3, 3)
cxy = mscohere(x, y, hann(8192), 4096);
plot(cxy)
hold on;
subplot(2, 3, 4)
cxy = mscohere(x, y, hann(4096), 3072);
plot(cxy)
hold on;
subplot(2, 3, 5)
cxy = mscohere(x, y, hann(4096), 1024);
plot(cxy)
hold on;
subplot(2, 3, 6)
cxy = mscohere(x, y, hann(4096), 0);
plot(cxy)
hold on;
%%}
%% frequency axis
[cxy, f] = mscohere(x, y, hann(2048), 1024, 512, EEG.srate); % window shape, (window length), overlap size, number of data for FFT, fs
figure;
plot(f, cxy)

%% adjacency matrix
CXY_mat = zeros(nbchan, nbchan);
chanPairs = [];
connectStrength = [];
chanPairs_sam = [];
connectStrength_sam = [];
chanPairs_bin= [];
CXY_mat_bin = zeros(nbchan, nbchan);

for i = 1:nbchan
    for j = i + 1:nbchan
        x = data(i, :);
        y = data(j, :);
        [cxy, f] = mscohere(x, y, hann(2048), 1024, 512, EEG.srate);
        CXY_mat(i, j) = mean(cxy(33:52));
        CXY_mat(j, i) = CXY_mat(i, j);
        chanPairs = [chanPairs; i j];
        connectStrength = [connectStrength, CXY_mat(i, j)];
        if (CXY_mat(i, j) > 0.8)
            chanPairs_sam = [chanPairs_sam; i j];
            connectStrength_sam = [connectStrength_sam, CXY_mat(i, j)];

            chanPairs_bin = [chanPairs_bin; i j];
            CXY_mat_bin(i, j) = 1;
            CXY_mat_bin(j, i) = 1;
        else
            CXY_mat_bin(i, j) = 0;
            CXY_mat_bin(j, i) = 0;
        end
    end
end

figure;
imagesc(CXY_mat);

% adjacency matrix to graph
ds.chanPairs = chanPairs;
ds.connectStrength = connectStrength;
figure;
topoplot_connect(ds, chanlocs, colormap);

% sampled adjacency matrix to graph
ds.chanPairs = chanPairs_sam;
ds.connectStrength = connectStrength_sam;
figure;
topoplot_connect(ds, chanlocs, colormap);

% binarized adjacency matrix to graph
for i = 1:nbchan
    for j = i + 1:nbchan
        x = data(i, :);
        y = data(j, :);
        [cxy, f] = mscohere(x, y, hann(2048), 1024, 512, EEG.srate);
        CXY_mat(i, j) = mean(cxy(33:52));
        if (CXY_mat(i, j) > 0.8)
            CXY_mat(i, j) = 1;
            CXY_mat(j, i) = 1;
            chanPairs_bin = [chanPairs_bin; i j];
        else
            CXY_mat(i, j) = 0;
            CXY_mat(j, i) = 0;
        end
    end
end

ds.chanPairs = chanPairs_bin;
figure;
topoplot_connect(ds, chanlocs, colormap)

% topoplot for degree
degree = sum(CXY_mat);
figure;
topoplot_connect(degree, chanlocs);
