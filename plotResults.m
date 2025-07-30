function plotResults(filename)
    if nargin < 1
        filename = 'log.csv';
    end
    
    if ~isfile(filename)
        error('File not found: %s', filename);
    end
    
    % Read CSV
    data = readtable(filename);
    
    % Check required columns
    requiredCols = {'Timestamp','Load','Demand','Action','Latency','Throughput','Reward'};
    if ~all(ismember(requiredCols, data.Properties.VariableNames))
        error('CSV does not have required columns');
    end

    % Plot 1: Latency
    figure;
    subplot(3,1,1);
    plot(data.Timestamp, data.Latency, '-o');
    title('Latency Over Time'); xlabel('Time'); ylabel('Latency (ms)'); grid on;

    % Plot 2: Throughput
    subplot(3,1,2);
    plot(data.Timestamp, data.Throughput, '-o');
    title('Throughput Over Time'); xlabel('Time'); ylabel('Throughput (Mbps)'); grid on;

    % Plot 3: Slice Allocation Histogram
    subplot(3,1,3);
    histogram(data.Action, 'BinMethod','integers');
    title('Slice Allocation Distribution'); xlabel('Slice Number'); ylabel('Count'); grid on;

    % Summary Metrics
    fprintf('\n✅ Summary Metrics:\n');
    fprintf('Average Latency: %.2f ms\n', mean(data.Latency));
    fprintf('Total Throughput: %.2f Mbps\n', sum(data.Throughput));
    fprintf('Average Reward: %.2f\n', mean(data.Reward));
end