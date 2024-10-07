clear; clc

loadVGON = load('VGON_degeneracy.mat');
fields = fieldnames(loadVGON);
for i = 1:length(fields)
    res = loadVGON.(fields{i});
    loss = res.loss;
    energy = res.energy;
    kl_div = res.kl_div;
    fidelity = res.fidelity; 
    disp(fields{i})
    fprintf('Loss: %.8f, Energy: %.8f, KL: %.8f, Fidelity: %.8f\n', ...
        loss, energy, kl_div, fidelity);
end
