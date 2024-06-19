%% SIIB Demo
% This script demonstrates how SIIB can be related to intelligibility scores.
% 
%--------------------------------------------------------------------------
% Copyright 2017: Steven Van Kuyk. 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%--------------------------------------------------------------------------
%
% Contact: steven.jvk@gmail.com
%
% References: 
%   [1] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An instrumental
%       intelligibility metric based on information theory', 2017
%   [2] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An evaluation of
%       intrusive instrumental intelligibility metrics', 2017
%
% The listening test data used in this demo is described in 
% [3] Kjems et al., 2009, 'Role of mask pattern in intelligibility of ideal binary-masked noisy speech.'
% The full data set is available at http://web.cse.ohio-state.edu/pnl/corpus/Kjems-jasa09/README.html
clc;
clear all;

% load speech and noise signal used in [3]
load audio_for_demo; % x is clean speech, n is noise

figure(1)
subplot(2,1,1); plot((1:length(x))/fs,x); title('clean speech')
xlabel('time, s')
subplot(2,1,2); plot((1:length(n))/fs,n); title('noise signal')
xlabel('time, s')

% listening test data
S50 = 15.1/100; % parameters for speech-shaped noise (see [3])
L50 = -7.3;
intelligibility = [0.01 20:20:80 99]' % percentage of words correct
SNRdB =  L50-log((100-intelligibility)./intelligibility) / (4*S50) % invert Kjems psychometric curve to find the required SNR [3]

% compute SIIB 
% takes 1-2 minutes if the C-code option in SIIB is selected (on a vanilla PC).
% takes 5-10 minutes if the MATLAB code option is selected
siib=zeros(size(intelligibility));
display('computing SIIB...'); tic
for i = 1:length(SNRdB)
    % clean speech
    x = sqrt(db2pow(SNRdB(i)))*x/std(x);
    
    % randomise noise segment
    start = randi([1,length(n)-length(x)-1]);
    finish = start+length(x)-1;
    n_seg = n(start:finish);
    n_seg = n_seg/std(n_seg);
    
    % distorted speech
    y=x+n_seg;
    
    siib(i) = SIIB(x,y,fs);
end
display('finished'); toc

% fit curve to data (the curve depends on what experimental procedures were used during the listening test (see [2]))
opts = optimset('Display','off');
x0 = [max(1/max(siib),1e-3),1]; % initial conditions
ab_best = lsqcurvefit(@(ab,d) 100*(1-exp(-ab(1)*d)).^ab(2) , x0, siib, intelligibility, [0.001,0.001], [], opts); % minimize the squared error
a = ab_best(1);
b = ab_best(2);
mapping_function = @(d) 100*(1-exp(-a*d)).^b;  % equation (13) in [1]

figure(2)
plot(siib, intelligibility,'x', 0:150, mapping_function(0:150))
xlabel('SIIB, b/s')
ylabel('intelligibility, % of words')
xlim([0,150])
ylim([0 100])