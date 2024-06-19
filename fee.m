% 数据
filters1 = {'ISFT', 'Matched filter', 'MVDR beamformer', 'SDW Wiener filter', 'Classical MW filter','MVDR beamformer-CS', 'SDW Wiener filter-CS', 'Classical MW filter-CS'};
sti1 = [0.9537, 0.9240, 0.9182, 0.8692, 0.8957,0.8718, 0.8702, 0.8957];
stoi1 = [0.8316, 0.7399, 0.7394, 0.7394, 0.7086,0.7394, 0.7399, 0.7091];



% 定义不同方法的颜色
colors = lines(max(length(filters1)));

% 绘制点状图
figure;
hold on;

% 绘制第一个表格的数据
for i = 1:length(filters1)
    scatter(sti1(i), stoi1(i), 100, colors(i,:), 'filled', 'DisplayName', filters1{i});
end



xlabel('STI');
ylabel('STOI');
title('Comparison of Evaluation Results');
legend('show', 'Location', 'northeastoutside');
grid on;
axis([0.86 0.96 0.7 0.85]);