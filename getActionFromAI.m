function action = getActionFromAI(kpiStruct)
    % Dummy AI decision logic
    if kpiStruct.latency < 20 && kpiStruct.throughput > 50
        action = 1;  % Allocate more resources
    else
        action = 0;  % No change
    end
end
