function plotstuff(type, src_path1, src_path2, dest_path)

switch type
    case 'precision'
        txt='precisions.txt';
        xdata=1:50;
        x_label='Location error threshold';
        y_label='Precision';
        plot_title='Precision Plot';
    case 'success'
        txt='successes.txt';
        xdata=0.02:0.02:1;
        x_label='Overlap threshold';
        y_label='Success rate';
        plot_title='Success Plot';
    otherwise
        fprintf('Invalid\n');
        return;
end

path='../LAIR_CLIF_PV/LAIR/AOI01';

if nargin < 4
    src_path1=fullfile(path, 'LAIR025', txt);
    src_path2=fullfile(path, 'LAIR050', txt);
    src_path3=fullfile(path, 'irrelevant', txt);
    dest_path=fullfile(path, sprintf('%s_plot.jpg', type));
end

data1=importdata(src_path1);
data2=importdata(src_path2);
data3=importdata(src_path3);

figure
plot(xdata, data1, 'r-', 'LineWidth', 2)
hold on;
plot(xdata, data2, 'b--', 'LineWidth', 2)
hold on;
plot(xdata, data3, 'g.', 'LineWidth', 2)
ylim([0 1])
xlabel(x_label), ylabel(y_label)
legend('LAIR025', 'LAIR050', 'ADNet original')
title(plot_title, 'FontSize', 24)
saveas(gcf, dest_path);