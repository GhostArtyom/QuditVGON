clear; clc

loadVGON = load('VGON_degeneracy.mat');
fields = fieldnames(loadVGON);
for i = 1:length(fields)
    res = loadVGON.(fields{i});
    loss = res.loss;
    state = res.state;
    energy = res.energy;
    kl_div = res.kl_div;
    fidelity = res.fidelity;
    disp(fields{i})
    fprintf('Loss: %.8f, Energy: %.8f, KL: %.8f, Fidelity: %.8f\n', ...
        loss, energy, kl_div, fidelity);
    choose = nchoosek(1:size(state, 1), 2);
    num = size(choose, 1);
    overlap = zeros(1, num);
    fprintf('Overlap: ')
    for j = 1:num
        ind = choose(j, :);
        s1 = state(ind(1), :).';
        s2 = state(ind(2), :).';
        overlap(j) = abs(s1' * s2)^2;
        fprintf('%.4e  ', overlap(j))
    end
    fprintf('\n\n')
end
