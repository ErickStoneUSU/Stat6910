function [x, y] = load_data()
a231control = append_data(dir('data/231 cell/control/*.txt'), 1, 0, 0, 0, 0);
a231negativecontrol = append_data(dir('data/231 cell/Negative control/*.txt'), 1, 0, 0, 0, 0);

a231gamma0 = append_data(dir('data/231 cell/IFNgamma/0/*.txt'), 1, 0, 1, 0, 0);
a231gamma10 = append_data(dir('data/231 cell/IFNgamma/10/*.txt'), 1, 0, 1, 0, 10);
a231gamma100 = append_data(dir('data/231 cell/IFNgamma/100/*.txt'), 1, 0, 1, 0, 100);

a231metformin2 = append_data(dir('data/231 cell/metformin/2mM/*.txt'), 1, 0, 0, 1, 2);
a231metformin4 = append_data(dir('data/231 cell/metformin/4mM/*.txt'), 1, 0, 0, 1, 4);
a231metformin6 = append_data(dir('data/231 cell/metformin/6mM/*.txt'), 1, 0, 0, 1, 6);
a231metformin8 = append_data(dir('data/231 cell/metformin/8mM/*.txt'), 1, 0, 0, 1, 8);
a231metformin16 = append_data(dir('data/231 cell/metformin/16mM/*.txt'), 1, 0, 0, 1, 16);

b549control = append_data(dir('data/A549/Cont/A549 Control/*.txt'), 0, 1, 0, 0, 0);
b549negativecontrol = append_data(dir('data/A549/NegCont/A549 negative control/*.txt'), 0, 1, 0, 0, 0);

b549gamma0 = append_data(dir('data/A549/A549 IFN gamma 0/*.txt'), 0, 1, 1, 0, 0);
b549gamma10 = append_data(dir('data/A549/A549 IFN gamma 10/*.txt'), 0, 1, 1, 0, 10);
b549gamma100 = append_data(dir('data/A549/A549 IFN gamma 100/*.txt'), 0, 1, 1, 0, 100);

b549metformin2 = append_data(dir('data/A549/A549 metformin/2mM/*.txt'), 0, 1, 0, 1, 2);
b549metformin4 = append_data(dir('data/A549/A549 metformin/4mM/*.txt'), 0, 1, 0, 1, 4);
b549metformin6 = append_data(dir('data/A549/A549 metformin/6mM/*.txt'), 0, 1, 0, 1, 6);
b549metformin8 = append_data(dir('data/A549/A549 metformin/8mM/*.txt'), 0, 1, 0, 1, 8);
b549metformin16 = append_data(dir('data/A549/A549 metformin/16mM/*.txt'), 0, 1, 0, 1, 16);
data = [a231control.' a231negativecontrol.' a231gamma0.' a231gamma10.' ...
    a231gamma100.' a231metformin2.' a231metformin4.' a231metformin6.' ...
    a231metformin8.' a231metformin16.' b549control.' b549negativecontrol.' ...
    b549gamma0.' b549gamma10.' b549gamma100.' b549metformin2.' b549metformin4.' ...
    b549metformin6.' b549metformin8.' b549metformin16.'].';
header = {'x' 'y' 'v1' 'v2' '' 'is_a231' 'is_b549' 'is_gamma' 'is_meta' 'amount'};
csvwrite()
x = data(:,1:4);
y = data(:,5:-1);
end

