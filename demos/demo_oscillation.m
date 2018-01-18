%This script will train and test a tanh firing rate model using the full-FORCE
%algorithm and compare it against FORCE learning
%Details are located here: https://arxiv.org/abs/1710.03070

clearvars;
global ah

%DTRLS: number of time steps between RLS updates
%dt: Euler timestep
%taux: decay time
%eta: strength of noise
p = struct('DTRLS',2,'dt',1e-3,'taux',1e-2,'eta',0);

%RLS: how many trials to do RLS training for
%test: how many trials to test for, after training
%init: number of trials for dynamics to settle from initial condition
T = struct('RLS',200,'test',50,'init',10);

%r: gain of recurrent connectivity
%fout: gain of output feedback
%fin: gain of input signal
g = struct('r',1.5,'fout',1.0,'fin',1.0);

%N: number of units
%out: dimension of output
%in: dimension of input
N = struct('N',300,'out',1,'in',1);

%% Figure

fh = figure('Color','w','Toolbar','none','Menubar','none');
ah = axes('LineWidth',1,'FontSize',8,'ylim',1.2 * [-1 1]);
xlabel('time (s)');

%% pick random connectivity

%J: random recurrent connectivity
%fout: fout into target-generating network
%fin: fin into target-generating and task-performing network

rng(1); %set random seed
ran = struct('J',1/sqrt(N.N) * randn(N.N),...
    'fout',-1 + 2 * rand(N.N,N.out),'fin',-1 + 2 * rand(N.N,N.in));

%% Train and test with full-FORCE and FORCE

V = eye(N.N); %use all of the modes of the target-generating netwwork
lrn = fF_v_F('train',g,N,p,ran,T.RLS,T.init,'osc',V);
ERR = fF_v_F('test', g,N,p,ran,T.test,T.init,'osc',V,lrn);

close(fh);
