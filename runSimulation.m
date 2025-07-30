function results = runSimulation(numSteps)
    % Default: 100 steps if not provided
    if nargin < 1
        numSteps = 100;
    end

    % Load environment
    env = NetworkSlicingEnv();

    % Load trained agent
    if exist('trainedDQNAgent.mat', 'file') ~= 2
        error('trainedDQNAgent.mat not found. Run trainAgent() first.');
    end
    data = load('trainedDQNAgent.mat');
    agent = data.agent;

    % Preallocate results
    results = struct();
    results.Step = (1:numSteps)';
    results.Slice = zeros(numSteps, 1);
    results.Throughput = zeros(numSteps, 1);
    results.Latency = zeros(numSteps, 1);

    % Reset environment
    obs = reset(env);

    for t = 1:numSteps
        % Get action from agent
        action = getAction(agent, {obs});
        if iscell(action)
            action = action{1};
        end

        % Step the environment
        try
            [nextObs, ~, ~, info] = step(env, action);
        catch ME
            warning('Error at step %d: %s', t, ME.message);
            break;
        end

        % Save data safely
        results.Slice(t) = action;

        % ✅ Defensive checks for info struct
        if isstruct(info)
            if isfield(info, 'Throughput')
                results.Throughput(t) = info.Throughput;
            else
                results.Throughput(t) = NaN;
            end

            if isfield(info, 'Latency')
                results.Latency(t) = info.Latency;
            else
                results.Latency(t) = NaN;
            end
        else
            results.Throughput(t) = NaN;
            results.Latency(t) = NaN;
        end

        % Move to next observation
        obs = nextObs;
    end

    % Save results
    save('simulationResults.mat', 'results');
    writetable(struct2table(results), 'simulation_results.xlsx');

    % === Plot 1: Throughput ===
    figure;
    plot(results.Step, results.Throughput, '-o', 'LineWidth', 2);
    xlabel('Step'); ylabel('Throughput (Mbps)');
    title('Throughput over Time'); grid on;

    % === Plot 2: Latency ===
    figure;
    plot(results.Step, results.Latency, '-o', 'LineWidth', 2);
    xlabel('Step'); ylabel('Latency (ms)');
    title('Latency over Time'); grid on;

    % === Plot 3: Slice Selection ===
    figure;
    plot(results.Step, results.Slice, '-o', 'LineWidth', 2);
    xlabel('Step'); ylabel('Slice Selected');
    title('Slice Selection over Time');
    yticks(unique(results.Slice)); grid on;
end
