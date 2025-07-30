function [initialObs, loggedSignals] = myResetFunction()
    initialObs = randn(3, 1);  % 3x1 column vector
    loggedSignals.State = initialObs;
end
