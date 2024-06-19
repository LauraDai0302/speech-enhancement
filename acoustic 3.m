% clc;close all; clear all;
% % Read audio files
% [audio1, fs] = audioread('clean_speech.wav'); % Replace with your audio file path
% [audio2, fs] = audioread('clean_speech_2.wav');
% [audio3, fs] = audioread('babble_noise.wav');
% [audio4, fs] = audioread('aritificial_nonstat_noise.wav');
% [audio5, fs] = audioread('Speech_shaped_noise.wav');
% 
% % Load impulse response file
% load('impulse_responses.mat'); % Assuming the impulse response file is named impulse_responses.mat
% %% unify all signal and noise to the same length with s1
% L = length(audio1);
% audio2 = [audio2; zeros(L-length(audio2),1)];
% audio3 = audio3(1:L);
% audio4 = audio4(1:L);
% audio5 = audio5(1:L);
% 
% % Define STFT parameters
% frame_duration = 0.02; % Frame duration (20 ms)
% frame_size = 320;  % Frame size (in samples)   was round(frame_duration * fs)
% overlap_fraction = 0.5; % Overlap fraction (40%)
% overlap_size = round(frame_size * overlap_fraction); % Overlap size (in samples)
% fft_size = frame_size; % FFT size
% 
% % Generate multi-microphone signals
% audio_signals = {audio1, audio2, audio3, audio4, audio5};
% impulse_responses = {h_target, h_inter1, h_inter2, h_inter3, h_inter4};
% 
% num_mics = 4;
% num_sources = length(audio_signals);
% mic_signals = cell(num_mics, num_sources);
% 
% % Convolve each audio signal with the corresponding impulse response
% for src = 1:num_sources
%     for mic = 1:num_mics
%         mic_signals{mic, src} = conv(audio_signals{src}, impulse_responses{src}(mic, :));
%     end
% end
% 
% % Perform STFT on each microphone signal
% stft_results = cell(num_mics, num_sources);
% 
% for mic = 1:num_mics
%     for src = 1:num_sources
%         signal = mic_signals{mic, src};
%         num_frames = floor((length(signal) - frame_size) / (frame_size - overlap_size))+1;
%         stft_matrix = zeros(num_frames, fft_size);
% 
%         for n = 1:num_frames
%             start_idx = (n - 1) * (frame_size - overlap_size) + 1;
%             end_idx = start_idx + frame_size - 1;
% 
%             frame = signal(start_idx:end_idx);
%             windowed_frame = frame .* hamming(frame_size);
%             fft_frame = fft(windowed_frame, fft_size);
%             stft_matrix(n,:) = fft_frame';
%         end
% 
%         stft_results{mic, src} = stft_matrix;
%     end
% end
% 
% % Find max time index
% [src_frames, ~] = size(stft_results{1, 1});
% max_frames = src_frames;
% % for mic = 1:num_mics
% %     for src = 1:num_sources    
% %         [frames, ~] = size(stft_results{mic, src});
% %         % update max_freq_bins 
% %         if frames > max_frames
% %             max_frames = frames;
% %         end
% %     end
% % end
% 
% % Generate the result matrix X,N
% X = cell(num_mics,1);
% N = cell(num_mics,1);
% for mic = 1:num_mics
%     % initialize
%     X{mic} = zeros(max_frames, fft_size);
%     N{mic} = zeros(max_frames, fft_size);
%     for src = 1:num_sources
%         % sum sources after STFT
%         i{mic} = zeros(max_frames, fft_size);
%         i{mic}(1:size(stft_results{mic, src},1),1:size(stft_results{mic, src},2)) = stft_results{mic, src};
%         X{mic} = X{mic} + i{mic};
% 
%         if src > 1
%             N{mic} = N{mic} + i{mic};
%         end
%     end
% end
% % 
% % 
% % % Example: Plot the STFT magnitude spectrum for the first audio signal at the first microphone
% % figure(1);
% % plot(abs(X{3}(1000,:)));
% % hold on;
% % plot(abs(stft_results{3,1}(1000,:)));
% % hold on;
% % plot(abs(stft_results{3,2}(1000,:)));
% % hold on;
% % plot(abs(stft_results{3,3}(1000,:)));
% % legend('mixed', 'target','clean_2','bubble');
% % xlabel('Frequency');
% % ylabel('Amplitude');
% % title('STFT Magnitude Spectrum for Audio 1, Microphone 1');
% % 
% %% Estimate A Rs Rn
% num_frames = size(X{1}, 1);  % Number of frames
% num_freq_bins = size(X{1}, 2);  % Number of frequency bins 
% 
% % Transform X to x
% x = zeros(src_frames, num_freq_bins, num_mics, 1);
% n = zeros(src_frames, num_freq_bins, num_mics, 1);
% for k = 1:src_frames
%     for l = 1:num_freq_bins
%         for m = 1: num_mics
%            x(k, l, m, 1) = X{m}(k,l);
%            n(k, l, m, 1) = N{m}(k,l);
%         end
%     end
% end
% 
% 
% % Initialize covariance matrices
% Rxx = zeros(src_frames, num_freq_bins, num_mics, num_mics);  % Signal covariance matrix
% Rnn = zeros(src_frames, num_freq_bins, num_mics, num_mics);  % Noise covariance matrix
% 
% % Estimate the covariance matrices
% alpha = 0.7;
% for l = 1:num_freq_bins
%     for k = 1:src_frames    
%         % Covariance of x
%         x_kl = squeeze(x(k, l, :, :));
%         n_kl = squeeze(n(k, l, :, :));
%         % n_power(k,l,:,:) = n_kl.*n_kl;
%         if k == 1
%             Rxx(k,l,:,:) = x_kl *x_kl'; 
%             Rnn(k,l,:,:) = n_kl *n_kl'; 
%         else
%             Rxx(k,l,:,:) = alpha * squeeze(Rxx(k-1,l,:,:)) + (1-alpha) *x_kl *x_kl';
%             Rnn(k,l,:,:) = alpha * squeeze(Rnn(k-1,l,:,:)) + (1-alpha) *n_kl *n_kl';
%         end
%         % % Covariance of n
%         % if k<=100
%         %     Rnn(k,l,:,:) = squeeze(Rnn(k,l,:,:))+ (1/100) * x_kl *x_kl';
%         % else 
%         %     Rnn(k,l,:,:) = Rnn(k,l,:,:);
%         % end
% 
%     end
% end
% 
% 
% % Initialize matrices for eigenvalues and eigenvectors
% eigenvalues = zeros(src_frames, num_freq_bins, num_mics);
% eigenvectors = zeros(src_frames, num_freq_bins, num_mics, num_mics);
% 
% % Compute the generalized eigenvalue decomposition for each time index and frequency bin
% for k = 1:src_frames
%     for l = 1:num_freq_bins 
%       [V, D] = eig(squeeze(Rxx(k, l, :, :)), squeeze(Rnn(k, l, :, :)));  % Generalized eigenvalue decomposition
%       % sort eigenvalues
%       [eigenvalues(k, l,:),idx] = sort(diag(D),'descend'); 
%       eigenvectors(k, l, :, :)  = V(:, idx); % sort
%     end
% end
% 
% lambda = eigenvalues(:, :, 1) -1; % sigmas^2
% % 
% % Get Q = inverse hermitian of U, a = q1
% Q = zeros(src_frames, num_freq_bins, num_mics, num_mics);
% for k = 1:src_frames
%     for l = 1:num_freq_bins 
%         U = squeeze(eigenvectors(k, l, :, :));
%         Q(k, l, :, :) = (inv(U'));
%     end
% end

%% Filter

scale = 1;
% w_mf = scale*a;
i = [1;0;0;0];
for k = 1:src_frames
    for l = 1:num_freq_bins 
        Rn = squeeze(Rnn(k, l, :, :));
        U = squeeze(eigenvectors(k, l, :, :));
        q = squeeze(Q(k, l, :, :));
        a_estimated = q(:,1);
        sigma2_s =  squeeze(lambda(k, l, :)); 
        Rs_estimated = sigma2_s * a_estimated *a_estimated';


        % ***Use cs to estimate ATF--paper method
        % a_cs = (squeeze(Rxx(k, l, :, :))-Rn)*i/(i'*(squeeze(Rxx(k, l, :,:))-Rn)*i); % this method also works
        % a_estimated = a_cs;



        % 1. Matched filter -- Minimize Output SNR
        w_kl_1 = 50*U(:,1);


        % 2. Classical Multi-channel Wiener filter, trade-off between noise and distortion, miu =1
        w_kl_2 = sigma2_s* inv(squeeze(Rxx(k, l, :, :)))*a_estimated;


        % 3. MVDR beamformer, distorsionless, miu =0
        w_kl_3 = inv(Rn)*(a_estimated)/(a_estimated'*inv(Rn)*a_estimated);

        % 4. signal-distortion weighted (SDW) Multichannel Wiener,  miu =0.5     
       
        miu = 0.5;
        w_kl_4 = sigma2_s/(sigma2_s + miu)*U(:,1)*q(:,1)'*i;

        x_kl = squeeze(x(k,l,:,:));

        s_hat_1(k,l) = w_kl_1' * x_kl;
        s_hat_2(k,l) = w_kl_2' * x_kl;
        s_hat_3(k,l) = w_kl_3' * x_kl;
        s_hat_4(k,l) = w_kl_4' * x_kl;
    end
end

% Initialize time-domain signal
N = (src_frames-1)*(frame_size - overlap_size) + frame_size;
x_time = zeros(N, 1);  % Allocate space for the time-domain signal
% Create a Hamming window for the iSTFT
window = hamming(frame_size, 'periodic');

% Perform the inverse STFT (iSTFT)
for k = 1:src_frames
    % Compute the inverse FFT for the current frame
    current_frame = s_hat_1(k, :);
    % current_frame = [current_frame, conj(current_frame(end-1:-1:2))]; % Mirror the frequency bins to get a full spectrum
    x_ifft = ifft(current_frame, fft_size)'; % Inverse FFT to get the time-domain frame

    % Apply the window
    x_ifft = real(x_ifft(1:frame_size)) .* window;

    % Overlap-add the frame to the signal
    start_index = (k-1) * (frame_size - overlap_size) + 1;
    x_time(start_index:start_index+frame_size-1) = x_time(start_index:start_index+frame_size-1) + x_ifft;
end

% Normalize the time-domain signal
x_time = x_time / max(abs(x_time));

% Plot the time-domain signal
figure;
plot(x_time);
xlabel('Time (samples)');
ylabel('Amplitude');
title('Time-Domain Signal');

% Save the time-domain signal as an audio file
output_filename = 'MSNR_recovered.wav';
audiowrite(output_filename, x_time, fs);
disp(['Audio file saved as ', output_filename]);


% Perform the inverse STFT (iSTFT)
x_time = zeros(N, 1);
for k = 1:src_frames
    % Compute the inverse FFT for the current frame
    current_frame = s_hat_2(k, :);
    current_frame = [current_frame, conj(current_frame(end-1:-1:2))]; % Mirror the frequency bins to get a full spectrum
    x_ifft = ifft(current_frame, fft_size)'; % Inverse FFT to get the time-domain frame

    % Apply the window
    x_ifft = real(x_ifft(1:frame_size)) .* window;

    % Overlap-add the frame to the signal
    start_index = (k-1) * (frame_size - overlap_size) + 1;
    x_time(start_index:start_index+frame_size-1) = x_time(start_index:start_index+frame_size-1) + x_ifft;
end

% Normalize the time-domain signal
x_time = x_time / max(abs(x_time));

% Plot the time-domain signal
figure;
plot(x_time);
xlabel('Time (samples)');
ylabel('Amplitude');
title('Time-Domain Signal');

% Save the time-domain signal as an audio file
output_filename = 'MW_recovered.wav';
audiowrite(output_filename, x_time, fs);
disp(['Audio file saved as ', output_filename]);


% Perform the inverse STFT (iSTFT)
x_time = zeros(N, 1);
for k = 1:src_frames
    % Compute the inverse FFT for the current frame
    current_frame = s_hat_3(k, :);
    current_frame = [current_frame, conj(current_frame(end-1:-1:2))]; % Mirror the frequency bins to get a full spectrum
    x_ifft = ifft(current_frame, fft_size)'; % Inverse FFT to get the time-domain frame

    % Apply the window
    x_ifft = real(x_ifft(1:frame_size)) .* window;

    % Overlap-add the frame to the signal
    start_index = (k-1) * (frame_size - overlap_size) + 1;
    x_time(start_index:start_index+frame_size-1) = x_time(start_index:start_index+frame_size-1) + x_ifft;
end

% Normalize the time-domain signal
x_time = x_time / max(abs(x_time));

% Plot the time-domain signal
figure;
plot(x_time);
xlabel('Time (samples)');
ylabel('Amplitude');
title('Time-Domain Signal');

% Save the time-domain signal as an audio file
output_filename = 'MVDR_recovered.wav';
audiowrite(output_filename, x_time, fs);
disp(['Audio file saved as ', output_filename]);



% Perform the inverse STFT (iSTFT)
x_time = zeros(N, 1);
for k = 1:src_frames
    % Compute the inverse FFT for the current frame
    current_frame = s_hat_4(k, :);
    current_frame = [current_frame, conj(current_frame(end-1:-1:2))]; % Mirror the frequency bins to get a full spectrum
    x_ifft = ifft(current_frame, fft_size)'; % Inverse FFT to get the time-domain frame

    % Apply the window
    x_ifft = real(x_ifft(1:frame_size)) .* window;

    % Overlap-add the frame to the signal
    start_index = (k-1) * (frame_size - overlap_size) + 1;
    x_time(start_index:start_index+frame_size-1) = x_time(start_index:start_index+frame_size-1) + x_ifft;
end

% Normalize the time-domain signal
x_time = x_time / max(abs(x_time));

% Plot the time-domain signal
figure;
plot(x_time);
xlabel('Time (samples)');
ylabel('Amplitude');
title('Time-Domain Signal');

% Save the time-domain signal as an audio file
output_filename = 'SDW_0.5_recovered.wav';
audiowrite(output_filename, x_time, fs);
disp(['Audio file saved as ', output_filename]);
