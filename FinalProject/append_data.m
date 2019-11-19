function [data] = append_data(loc, is_a231, is_b549, is_gamma, is_meta, amount)
% a231 b549 is_gamma is_meta 
data = [];
label = [is_a231 is_b549 is_gamma is_meta amount];
for i = 1:length(loc)
    d = csvread([loc(i).folder '\' loc(i).name]);
    e = [cell2mat(struct2cell(datastats(d))).' label];
    data = [data [e.']];
end
data = data.';
end

