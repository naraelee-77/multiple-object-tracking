function net = modify_dagnn(dagnn_net, id)

net = dagnn_net;
for i = 1:numel(net.vars)
    net.vars(i).name = sprintf('%s_id%d', net.vars(i).name, id);
end
for i = 1:numel(net.params)
    net.params(i).name = sprintf('%s_id%d', net.params(i).name, id);
end
for i = 1:numel(net.layers)
    net.layers(i).name = sprintf('%s_id%d', net.layers(i).name, id);
    net.layers(i).inputs{1} = sprintf('%s_id%d', net.layers(i).inputs{1}, id);
    net.layers(i).outputs{1} = sprintf('%s_id%d', net.layers(i).outputs{1}, id);
%     net.layers(i).inputIndexes = net.layers(i).inputIndexes + numel(net.vars)*(id-1);
%     net.layers(i).outputIndexes = net.layers(i).outputIndexes + numel(net.vars)*(id-1);
    if ~isempty(net.layers(i).params)
        for j = 1:numel(net.layers(i).params)
            net.layers(i).params{j} = sprintf('%s_id%d', net.layers(i).params{j}, id);
%             net.layers(i).paramIndexes(j) = net.layers(i).paramIndexes(j) + numel(net.params)*(id-1);
        end
    end
end