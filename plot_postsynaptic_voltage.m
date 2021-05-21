function plot_postsynaptic_voltage(t, sample, K, W, TH, tau_m, sample_color)
%PLOT_POSTSYNAPTIC_VOLTAGE Evaluate and plot the post-synaptic voltage in
%response to a given input
%   t               - a times vector
%   sample          - the input neurons' spike times
%   K               - the voltage kernel function
%   W               - the synaptic weights
%   TH              - the neuron's voltage threshold
%   tau_m           - the neuron's membrane time constant
%   sample_color    - the plot's color

t_final = t(end);

% Get the data for the voltage plot
V = zeros(size(t));   % voltage
for i = 1:length(sample)        % loop over neurons
    for j = 1:length(sample{i})	% loop over spikes
        t_i_j	= sample{i}(j);           % spike time
        V       = V + W(i).*K(t - t_i_j);	% unresetted voltages
    end
end

% Add the threshold mechanism to the voltage
% NOTE: The implemented threshold mechanism is different than the one seen 
%       in class, yet it results in very similar outputs. It was chosen due 
%       to technical reasons, don't overthink it.
spk_ker = @(x) (x > 0).*exp(-x./tau_m);
for t_idx = 1:length(t)
    if V(t_idx) > TH  % threshold crossing
        V(t_idx)	= 1.5*TH;                           % for vizualisation
        V           = V - TH.*spk_ker(t - t(t_idx));	% reset future voltage
    end
end

plot(t.*1000, V.*1000, ...
    'Color', sample_color);
hold on;
plot([0, t_final].*1000, [TH, TH].*1000, 'k--');
hold off;
xlim([0, t_final].*1000);
ylim([-0.5*TH, 1.5*TH].*1000);
xlabel('time [ms]');
ylabel('Voltage [mV]');

end

