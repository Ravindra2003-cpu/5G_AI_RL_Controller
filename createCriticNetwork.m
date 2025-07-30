function critic = createCriticNetwork()

    % Observation आणि Action Specification define कर
    obsInfo = rlNumericSpec([3 1]);          % Observation: 3x1 vector
    obsInfo.Name = 'NetworkState';

    actionInfo = rlFiniteSetSpec([1 2 3]);   % Discrete Actions: 1, 2, 3
    actionInfo.Name = 'ResourceAction';

    % Critic Network बनवणे (Fully Connected Layers)
    statePath = [
        featureInputLayer(3, 'Name', 'state')           % Input layer
        fullyConnectedLayer(32, 'Name', 'fc1')          % Hidden layer 1
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(32, 'Name', 'fc2')          % Hidden layer 2
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(3, 'Name', 'output')        % Output layer (3 actions)
    ];

    % Deep Learning Network तयार कर
    criticNet = dlnetwork(layerGraph(statePath));

    % Representation Options define कर
    criticOpts = rlRepresentationOptions( ...
        'LearnRate', 1e-3, ...
        'GradientThreshold', 1);

    % Q-value Representation तयार कर
    critic = rlQValueRepresentation( ...
        criticNet, ...
        obsInfo, ...
        actionInfo, ...
        'Observation', {'state'}, ...
        criticOpts);

end
