% aer1517_a1_3: Main script for Problem 1.3 Approximate Dynamic 
%               Programming.
%
% adapted from: Borrelli, Francesco: "ME 231 Experiential Advanced Control
% Design I"
%
% --
% Control for Robotics
% AER1517 Spring 2022
% Assignment 1
%
% --
% University of Toronto Institute for Aerospace Studies
% Dynamic Systems Lab
%
% Course Instructor:
% Angela Schoellig
% schoellig@utias.utoronto.ca
%
% Teaching Assistants: 
% SiQi Zhou: siqi.zhou@robotics.utias.utoronto.ca
% Adam Hall: adam.hall@robotics.utias.utoronto.ca
% Lukas Brunke: lukas.brunke@robotics.utias.utoronto.ca
%
% --
% Revision history
% [22.01.17, LB]    first version
% [22.01.24, LB]    updated horizon and initial state
% Modified and completed by Francisco Granda

clear all
close all
clc

%% set up system

% inverted pendulum parameters
l = 1.0; % length
g = 9.81; % gravitational constant
m = 1.0; % mass

% create inverted pendulum system
sys = InvertedPendulum(l, g, m);

% controller parameters
Q = diag([1, 0.1]);
R = 1;
N = 25;

% linearization point
x_up = [pi; 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
CODE TO USE SYMBOLIC VARIABLES FOR THE LINEARIZATION AND DISCRETIZATION
PROCESS. THIS CODE OUTPUTS MATRICES A_d AND B_d. BOTH MATRICES WERE
STORED IN "Sys_d.mat", SO THIS CODE WAS RUN ON A SEPARATE SCRIPT.
syms x_1 x_2 x_dot_1 x_dot_2 u
syms M g l
I = (M*l^2)
% Compute partial derivatives
eqn1 = -x_dot_1 + x_2
eqn2 = -x_dot_2 - M*g*l*sin(x_1)/I + u/I
simplify(eqn2)
eqns = [eqn1;eqn2]
A_c = jacobian(eqns,[x_1,x_2])
B_c = jacobian(eqns,[u])
% Evaluate with point (x1',x2',u') = (pi, 0, 0)
A_c = subs(A_c,[x_1,x_2,u],[pi,0,0])
B_c = subs(B_c,[x_1,x_2,u],[pi,0,0])
% Evaluate constant values
A_c = double(subs(A_c,[g,l],[9.81,1]))
B_c = double(subs(B_c,[M,l],[1,1]))
sys_c = ss(A_c,B_c,zeros(2),0);
sys_d = c2d(sys_c,0.1);
A_d = sys_d.A
B_d = sys_d.B
%}
%LOAD DISCRETIZED MATRICES
load Sys_d.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% cost functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
terminal_cost = @(x) [x(1),x(2)] * Q * [x(1);x(2)];
stage_cost = @(x,u) [x(1),x(2)] * Q * [x(1);x(2)] + R * u^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate optimal control using dynamic programming and gridding

% grid state-space
num_points_x1 = 10;
num_points_x2 = 5;
X1 = linspace(-pi/4, pi/4, num_points_x1);
X2 = linspace(-pi/2, pi/2, num_points_x2);

% allocate arrays for optimal control inputs and cost-to-go 
U = zeros(num_points_x1, num_points_x2);
J = zeros(num_points_x1, num_points_x2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation of DPA
syms u
for i = N:-1:1
    for j = 1:size(X1,2)
        for k = 1:size(X2,2)
            if i == N
                x_jk = [X1(j);X2(k)];
                J_jk = terminal_cost(x_jk);
                J(j,k) = J_jk;
                state{j,k} = x_jk;
            else 
                x_jk = state{j,k};
                x_jk_next = A_d*x_jk +B_d*u;
                J_jk = J(j,k) + R*u^2 + stage_cost(x_jk_next,u);
                J_jk = matlabFunction(J_jk);
                [u_jk,fval] = fminunc(J_jk,0);
                J_jk = J_jk(u_jk);
                U(j,k) = u_jk;
                J(j,k) = J_jk;
                state{j,k} = double(subs(x_jk_next,u_jk));
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plot optimal control and cost-to-go
figure
subplot(1, 2, 1)
surf(X1, X2, U')
xlabel('x_1')
ylabel('x_2')
zlabel('u')
subplot(1, 2, 2)
surf(X1, X2, J')
xlabel('x_1')
ylabel('x_2')
zlabel('J')

%% apply control law and simulate inverted pendulum
% create the controlled inverted pendulum system
control_sys = InvertedPendulum(l, g, m, X1, X2, U, x_up);

% initial condition
x0 = x_up + [-pi/6; 0];

% duration of simulation
t = [0, 10];

% simulate control system
[t, x] = ode45(@control_sys.controlled_dynamics, t, x0);

% determine control inputs from trajectory
u = zeros(size(t));
for i = 1 : length(t)
    u(i) = control_sys.mu(x(i, :)' - x_up);
end

%% plot state and input trajectories
figure
subplot(2, 1, 1)
hold on
plot(t, x(:, 1))
plot(t, x(:, 2))
xlabel('t')
ylabel('x_1 and x_2')
hold off
legend('\theta','d\theta/dt')
grid on
subplot(2, 1, 2)
plot(t, u)
xlabel('t')
ylabel('u')
grid on