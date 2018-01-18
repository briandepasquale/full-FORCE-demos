function varargout = fF_v_F(mode,g,N,p,ran,TRLS,Tinit,task,V,varargin)
%Inputs:
%mode: for training, or testing
%g: various gain parameters
%N: network size parameters
%p: an eclectic group of parameters, related to time, etc.
%TRLS: how to long to do RLS training for
%Tinit: how long to drive the system before learning
%task: name of the task
%V: eigenvectors for the target-generating network, (possible) learned in a previous function

%Output: 
%mode == train: learned parameters from RLS
%mode == test: nMSError

global ah

%% Unpack varargin

if numel(varargin) > 0
    lrn = varargin{1};
end

%% Unpack parameters stored in struct

gr = g.r; %gain of J
gfout = g.fout; %gain on output inputs
gfin = g.fin; %gain on inputs
taux = p.taux; %timescale
dt = p.dt; %euler timesteps
eta = p.eta; %noise
etaux = exp(-dt/taux); %timescale

NN = N.N; %number of units
Nout = N.out; %number of

J = ran.J; %recurrence for target-generating network
ufout = ran.fout; %"output" input matrix
ufin = ran.fin; %input matrix

DTRLS = p.DTRLS; %number of timesteps between RLS updates

%% Line handles for plotting data

lh(1) = line(ah,'Color','r','LineWidth',1,'Marker','none','LineStyle','-');
lh(2) = line(ah,'Color','b','LineWidth',1,'Marker','none','LineStyle','-');
lh(3) = line(ah,'Color','k','LineWidth',1,'Marker','none','LineStyle','--');
lh(4) = line(ah,'Color','k','LineWidth',1,'Marker','none','LineStyle','--');

legend('Z_{fF}','Z_{F}','f_{out}','f_{in}');legend boxoff

%% Initalize various parameters for RLS

Nn_fF = size(V,2); %dimension of learned targets for fF network

if strcmp(mode,'train')

    w_fF = zeros(Nn_fF,NN); %learned recurrent matrix for fF network
    W_fF = zeros(Nout,NN); %output matrix for fF
    PS_fF = 1 * eye(NN); %inverse covariance matrix for fF
    
    W_F = zeros(Nout,NN); %output matrix for F
    PS_F = 1 * eye(NN); %inverse covariance matrix for F
    
elseif strcmp(mode, 'test')
    
    w_fF = lrn.w_fF; %learned recurrent matrix for fF
    W_fF = lrn.W_fF; %output matrix for fF
    W_F = lrn.W_F; %output matrix for F
    
end

%% Initalize static parameters and network states

Jr = gr * J; %scaled recurrent of target-generating network
uFout = gfout * ufout; %scaled "output" input
uFin = gfin * ufin; %scaled input

x_teach = 1e0 * randn(NN,1); %target-generating network state
x_learn_fF = x_teach; %full-FORCE task-performing network state
x_learn_F = x_learn_fF; %same, but for FORCE

z_fF = zeros(Nn_fF,1); %state vector of learned dyanamics, full-FORCE network
Z_F = zeros(Nout,1); %output of FORCE network
r_F = tanh(x_learn_F); %rates of FORCE network

ttrial = inf; %time index, for calculating the end of a trial
TTrial = 0; %length of a trial, will change within the simulation loop
EZ_fF = 0; %used to compute running update of Error variance, full-FORCE
NF_fF = 0; %same, but target variance
EZ_F = 0; %same, but for FORCE
NF_F = 0;

ttime = 0; %counts the number of trials completed so far, to terminate the simulation loop

%% Run simulation

while ttime <= TRLS+Tinit
    
    if ttrial > TTrial %new trial starts, generate new inputs and outputs
        
        ttrial = 1; %start counter for elapsed time in a trial
        
        switch task
            case 'osc'
                [fin,fout,TTrial] = trial_osc(p);
        end
        
        %compute the inputs to each unit
        Fout = uFout * fout;
        Fin = uFin * fin;
        
        Zs_fF = zeros(Nout,TTrial); %to accumulate generated outputs, for plotting
        Zs_F = zeros(Nout,TTrial);       
        
        ttime = ttime + 1; %increment trial counter by 1
        
    end
    
    %target generating network
    f_fF = Jr * tanh(x_teach) + Fout(:,ttrial);
    xinf = f_fF + Fin(:,ttrial);
    x_teach = xinf + (x_teach - xinf) * etaux;
    f_fF = V' * f_fF;
    
    %full-FORCE task-performing network
    x_learn_fF = x_learn_fF + (dt/taux) * (-x_learn_fF + ...
        (V * z_fF + Fin(:,ttrial))) + sqrt(eta * dt) * randn(NN,1);
    r_fF = tanh(x_learn_fF);
    
    %FORCE task-performing network
    x_learn_F = x_learn_F + (dt/taux) * (-x_learn_F + ...
        (Jr * r_F + uFout * Z_F + Fin(:,ttrial))) + sqrt(eta * dt) * randn(NN,1);
    r_F = tanh(x_learn_F);
    
    %compute the feedback from learning
    if ttime > Tinit
        Z_F = W_F * r_F;
        z_fF = w_fF * r_fF;
        Z_fF = W_fF * r_fF;
    %if initialization period hasn't elapsed, used the target
    else
        z_fF = f_fF;
        Z_fF = fout(:,ttrial);
        Z_F = fout(:,ttrial);
    end
    
    %save the generated output for plotting
    Zs_fF(:,ttrial) = Z_fF;
    Zs_F(:,ttrial) = Z_F;
    
    %do RLS
    if rand < 1/DTRLS && ttime > Tinit && strcmp(mode,'train')
        
        %fF
        xP = PS_fF * r_fF;
        k = (1 + r_fF' * xP)\xP';
        
        PS_fF = PS_fF - xP * k;
        w_fF = w_fF - (z_fF - f_fF) * k;
        W_fF = W_fF - (Z_fF - fout(:,ttrial)) * k;
        
        %F
        xP = PS_F * r_F;
        k = (1 + r_F' * xP)\xP';
        
        PS_F = PS_F - xP * k;
        W_F = W_F - (Z_F - fout(:,ttrial)) * k;
        
    end
    
    
    % text output, plot things, and flush gathered data
    if ttrial == TTrial
             
        clc;
        fprintf('%s fullFORCE \nfullFORCE Error: %g\nForce: %g\n%g trials of %g \n', ...
            mode, 100 * EZ_fF/NF_fF, 100 * EZ_F/NF_F, ttime, (TRLS+Tinit));
        
        set(lh(1),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',Zs_fF(1,:));
        set(lh(2),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',Zs_F(1,:));
        set(lh(3),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',fout(1,:));
        set(lh(4),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',fin(1,:));
        
        drawnow;
        
    end
    
    %compute running value of normalized error
    if ttime > Tinit
        
        EZ_fF = EZ_fF + (Z_fF - fout(:,ttrial))' * (Z_fF - fout(:,ttrial));
        NF_fF = NF_fF + fout(:,ttrial)' * fout(:,ttrial);
        
        EZ_F = EZ_F + (Z_F - fout(:,ttrial))' * (Z_F - fout(:,ttrial));
        NF_F = NF_F + fout(:,ttrial)' * fout(:,ttrial);
        
    end
    
    ttrial = ttrial + 1; %increment trial time counter
    
end

if strcmp(mode,'train')
    
    %save learned matrices to structs
    lrn.w_fF = w_fF;
    lrn.W_fF = W_fF;
    lrn.PS_fF = PS_fF;
    lrn.W_F = W_F;
    lrn.PS_F = PS_F;
    
    varargout{1} = lrn;
    
elseif strcmp(mode,'test')
    
    %final error
    ERR(1)= 100 * EZ_fF/NF_fF;
    ERR(2) = 100 * EZ_F/NF_F;
    
    varargout{1} = ERR;
    
end

delete(lh); %remove old line handles

