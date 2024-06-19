clc;close all; clear all;
%% Evaluation

% 示例：计算相干语音可懂度指数（CSII）

% 加载语音信号
[speech, fs] = audioread('clean_speech.wav');  % 加载干净的语音信号




% 列出所有符合通配符 '*.wav' 的文件
allFiles = dir('*.wav');

% 筛选文件名中包含 '240' 的文件
selectedFiles = {};
for k = 1:length(allFiles)
    if contains(allFiles(k).name, 'recovered')
        selectedFiles{end+1} = fullfile(allFiles(k).folder, allFiles(k).name);
    end
end

% STI metric
ds_sti = fileDatastore(selectedFiles, 'ReadFcn', @(filename) calcSTI(filename, speech));  
stis = readall(ds_sti);
leg = legend('show');
leg.Interpreter = 'none';
leg.Location = 'southoutside';
ylabel('Audio Signal')
xlabel('Samples')
STIs = table(ds_sti.Files, stis, 'VariableNames', {'Filename', 'STI'})



% STOI metric
ds_stoi = fileDatastore(selectedFiles, 'ReadFcn', @(filename) calcSTOI(filename, speech));  
stis = readall(ds_stoi);
STOIs = table(ds_stoi.Files, stis, 'VariableNames', {'Filename', 'STOI'})




function thisSTI = calcSTI(filename, speech)
% Gets STI given audio filename    
    info = audioinfo(filename);
    fs = info.SampleRate;
    data = audioread(filename);
    [thisSTI,~] = stipa(data, fs, speech, fs);
    hold on
    plot(data, 'DisplayName', filename)
end




function thisSTOI = calcSTOI(filename, speech)
% Gets STOI given audio filename    
    info = audioinfo(filename);
    fs = info.SampleRate;
    data = audioread(filename);
    len = min(length(speech),length(data));
    thisSTOI = stoi(data(1:len),speech(1:len),fs);
    hold on
    plot(data, 'DisplayName', filename)
end