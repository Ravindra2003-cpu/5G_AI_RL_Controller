% AI Controller for Slice Allocation using Deep Q-Learning
% Requires Reinforcement Learning Toolbox

classdef NetworkSlicingEnv < rl.env.MATLABEnvironment
    properties
        % Define environment parameters
        CurrentLoad = 0;      % Current network load
        UserDemand = 0;       % Current user demand
        NumSlices = 5;        % Number of available slices
        MaxLatency = 100;     % Maximum allowed latency (ms)
        
        % Slice allocation parameters
        SliceCapacity = [100 120 150 200 250]; % Capacity per slice
        AllocatedSlices = zeros(1,5);          % Track allocations
        
        % Performance metrics
        TotalLatency = 0;
        TotalThroughput = 0;
    end
    
    methods
        % Constructor
        function this = NetworkSlicingEnv()
            ObservationInfo = rlNumericSpec([3 1]); % Load, Demand, Timestep
            ObservationInfo.Name = 'Network State';
            ObservationInfo.Description = 'Load, UserDemand, Timestep';
            
            ActionInfo = rlFiniteSetSpec(1:5); % Slice numbers 1-5
            ActionInfo.Name = 'Slice Selection';
            
            this = this@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
        end
        
        % Reset environment
        function InitialObservation = reset(this)
            % Initialize environment state
            this.CurrentLoad = randi([0 100]);
            this.UserDemand = randi([10 250]);
            this.AllocatedSlices = zeros(1,5);
            this.TotalLatency = 0;
            this.TotalThroughput = 0;
            
            InitialObservation = [this.CurrentLoad; this.UserDemand; 0];
        end
        
        % Step function
        function [Observation, Reward, IsDone, LoggedSignals] = step(this, Action)
            % Check validity of action
            Action = floor(Action);
            if Action < 1 || Action > this.NumSlices
                error('Invalid slice selection');
            end
            
            % Update state
            this.AllocatedSlices(Action) = this.AllocatedSlices(Action) + 1;
            
            % Calculate latency and throughput
            sliceLoad = this.AllocatedSlices(Action);
            sliceCapacity = this.SliceCapacity(Action);
            
            % Simple latency model (should be replaced with your actual model)
            latency = 10 + (90 * sliceLoad / sliceCapacity);
            throughput = min(this.UserDemand, sliceCapacity - sliceLoad);
            
            % Store metrics
            this.TotalLatency = this.TotalLatency + latency;
            this.TotalThroughput = this.TotalThroughput + throughput;
            
            % Calculate reward (negative latency, positive throughput)
            Reward = throughput/10 - latency/10;
            
            % Generate new random demand and load for next step
            this.CurrentLoad = randi([0 100]);
            this.UserDemand = randi([10 250]);
            
            % Create observation vector
            Observation = [this.CurrentLoad; this.UserDemand; 0];
            
            % Training runs for 100 steps
            IsDone = rem(this.TotalLatency, 100) == 0;
            
            % No specific signals to log
            LoggedSignals = [];
        end
    end
end

% Main Training Script
function trainSliceAllocationAgent()
    % Create the environment
    env = NetworkSlicingEnv();
    
    % Create a deep Q-network (DQN) agent
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);
    
    % Define the neural network layers
    layers = [
        featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'output')
    ];
    
    % Define the network options
    net = dlnetwork(layers);
    
    % Create the critic representation
    critic = rlVectorQValueFunction(net, obsInfo, actInfo);
    
    % Define agent options
    agentOpts = rlDQNAgentOptions(...
        'SampleTime', 1,...
        'UseDoubleDQN', true,...
        'TargetUpdateMethod', 'smooth',...
        'TargetUpdateFrequency', 100,...
        'ExperienceBufferLength', 10000,...
        'DiscountFactor', 0.99,...
        'MiniBatchSize', 128);
    
    % Create the DQN agent
    agent = rlDQNAgent(critic, agentOpts);
    
    % Training options
    trainOpts = rlTrainingOptions(...
        'MaxEpisodes', 1000,...
        'MaxStepsPerEpisode', 100,...
        'ScoreAveragingWindowLength', 10,...
        'SaveAgentCriteria', "EpisodeReward",...
        'SaveAgentValue', 100,...
        'SaveAgentDirectory', 'trainedAgents');
    
    % Train the agent
    trainingStats = train(agent, env, trainOpts);
    
    % Save the trained agent
    save('trainedSliceAllocationAgent.mat', 'agent');
end

% Function to generate allocation table using trained agent
function allocationTable = generateAllocationTable(agent, numPredictions)
    env = NetworkSlicingEnv();
    observation = reset(env);
    allocationTable = zeros(numPredictions, 4); % [Step, Load, Demand, Allocation]
    
    for step = 1:numPredictions
        action = getAction(agent, observation);
        allocationTable(step,:) = [step, observation(1), observation(2), action];
        [observation, ~, ~, ~] = step(env, action);
    end
    
    % Display the allocation table
    disp('Slice Allocation Table:');
    disp('Step | Load | Demand | AllocatedSlice');
    disp(allocationTable);
    
    % Calculate and display performance metrics
    avgLatency = env.TotalLatency/numPredictions;
    avgThroughput = env.TotalThroughput/numPredictions;
    
    fprintf('\nPerformance Metrics:\n');
    fprintf('Average Latency: %.2f ms\n', avgLatency);
    fprintf('Average Throughput: %.2f Mbps\n', avgThroughput);
    fprintf('Slice Utilization:\n');
    for i = 1:5
        fprintf('Slice %d: %d allocations\n', i, sum(allocationTable(:,4) == i));
    end
end