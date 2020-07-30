function z = EnKF_step(z, F, N, d, b, R)
% EnKF_step A single step of the ensemble Kalman Filter.
%
% z is the initial ensemble of particles of size (d+b)-by-N.
% F is the forward operator.
% N is the number of particles in the ensemble.
% d is the dimension of the state u.
% b is the dimension of the data.
% R is the noise covariance.

% Prediction step.
for n=1:N
    u = z(1:d,:);       % Get the state.
    Fu = F(u);          % Update the state with the forward operator.
    z(d+1:d+b,:) = Fu;  % Get the data.
end

z_bar = mean(z, 2);
C = cov(z', 1); % Normalize by N

% Analysis step.
H = zeros(d+b,b);
H(d+1:d+b,:) = eye(b); % Projection operator.
for n=1:N
    y = z(d+1:d+b,:) + mvnrnd('SIGMA', R); % Observed data.
    yHz = y - H*z(:,n);
    KyHz = C*H'*((H*C*H' + R)\yHz);
    z(:,n) = z(:,n) + KyHz;
end


end

