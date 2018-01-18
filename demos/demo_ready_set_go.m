%This script will train and test a rate model using the full-FORCE
%algorightm. It will train it once using hints, and once without using
%hints on the "ready-set-go" task.

clearvars;

global ah

%DTRLS: number of time steps between RLS updates
%dt: timestep
%taux: decay time
p = struct('DTRLS',1,'dt',1e-3,'taux',1e-2,'eta',0.0);

%RLS: how long to do RLS training for
%test: how long to test for, after training
%init: length of time for dynamics to settle from initial condition
T = struct('RLS',1000,'test',500 ,'init',10,'data',100);

%r: gain of recurrent connectivity
%fout: gain of output feedback
%fin: gain of input signal
%fhint: hint gain
g = struct('r',1.5,'fout',1.0,'fin',1.0,'fhint',1.0);

%N: number of units
%out: dimension of output
%in: dim of input
%hint: dim of hint
N = struct('N',1000,'out',1,'in',2,'hint',1);

%% Figure

fh = figure('Color','w','Toolbar','none','Menubar','none');
ah = axes('LineWidth',2,'FontSize',16,'ylim',[-0.6 2.1]);
xlabel('time (s)');

%% random connectivity

rng(1);
ran = struct('J',1/sqrt(N.N) * randn(N.N),'fout',(-1 + 2 * rand(N.N,N.out)),...
    'fin',-1 + 2 * rand(N.N,N.in),'fhint',-1 + 2 * rand(N.N,N.hint));

%% Train and test with full-FORCE

V = eye(N.N);
lrn = fullFORCE('train',g,N,p,ran,T.RLS,T.init,'ready_set_go','nohint',V);
ERR = fullFORCE('test', g,N,p,ran,T.test,T.init,'ready_set_go','nohint',V,lrn);

close(fh);
