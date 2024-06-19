% Load the SIIB function
SIIB_function = @SIIB;

% Read the original and recovered audio files
[original_signal, fs_original] = audioread('clean_speech.wav'); 
% [recovered_signal, fs_recovered] = audioread('MF_0.75.wav');
% [recovered_signal1, fs_recovered1] = audioread('MVDR_0.75.wav');
% [recovered_signal2, fs_recovered2] = audioread('MW_0.75.wav');
% [recovered_signal3, fs_recovered3] = audioread('SDW_0.75.wav');
[recovered_signal4, fs_recovered4] = audioread('CS_MVDR_recovered 1.wav');

% Trim the signals to the same length
min_length = min(length(original_signal), length(recovered_signal4));
original_signal = original_signal(1:min_length);

% recovered_signal = recovered_signal(1:min_length);
% recovered_signal1 = recovered_signal1(1:min_length);
% recovered_signal2 = recovered_signal2(1:min_length);
% recovered_signal3 = recovered_signal3(1:min_length);
recovered_signal4 = recovered_signal4(1:min_length);

% 
% % Check if the sampling rates match
% if fs_original ~= fs_recovered
%     error('The sampling rates of the original and recovered signals do not match.');
% end
% length(recovered_signal)
% length(original_signal)
% Calculate SIIB
% I1 = SIIB_function(original_signal, recovered_signal, fs_original);
% I2 = SIIB_function(original_signal, recovered_signal1, fs_original);
% I3 = SIIB_function(original_signal, recovered_signal2, fs_original);
% I4 = SIIB_function(original_signal, recovered_signal3, fs_original);
I5 = SIIB_function(original_signal, recovered_signal4, fs_original);
% % Display the result
% fprintf('The Speech Intelligibility in Bits (SIIB) MF0.75 is: %.2f b/s\n', I1);
% fprintf('The Speech Intelligibility in Bits (SIIB) MVDR0.75 is: %.2f b/s\n', I2);
% fprintf('The Speech Intelligibility in Bits (SIIB) cmw0.75 is: %.2f b/s\n', I3);
% fprintf('The Speech Intelligibility in Bits (SIIB) sdw0.75 is: %.2f b/s\n', I4);
fprintf('The Speech Intelligibility in Bits (SIIB) CS_MVDR is: %.2f b/s\n', I5);