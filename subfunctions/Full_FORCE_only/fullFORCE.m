function varargout = fullFORCE(mode,g,N,p,ran,TRLS,Tinit,task,hint,V,varargin)
%Inputs:
%mode: for training, or testing
%g: various gain parameters
%N: network size parameters
%p: an eclectic group of parameters, related to time, etc.
%TRLS: how to long to do RLS training for
%Tinit: how long to drive the system before learning
%task: name of the task
%hint: use a hint or not
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
gfhint = g.fhint; %gain on hint
taux = p.taux; %timescale
dt = p.dt; %euler timesteps
eta = p.eta; %noise
etaux = exp(-dt/taux); %timescale

NN = N.N; %number of units
Nout = N.out; %number of outputs

J = ran.J; %recurrence for target-generating network
ufout = ran.fout; %"output" input matrix
ufin = ran.fin; %input matrix
ufhint = ran.fhint; %hint matrix

DTRLS = p.DTRLS; %number of timesteps between RLS updates

%% Line handles for plotting data

lh(1) = line(ah,'Color','r','LineWidth',1,'Marker','none','LineStyle','-');
lh(2) = line(ah,'Color','k','LineWidth',1,'Marker','none','LineStyle','--');
lh(3) = line(ah,'Color','k','LineWidth',1,'Marker','none','LineStyle','--');
lh(4) = line(ah,'Color','k','LineWidth',1,'Marker','none','LineStyle','--');

legend('Z','f_{hint}','f_{out}','f_{in}');legend boxoff

%% Initalize various parameters for RLS

Nn = size(V,2); %dimension of learned targets

if strcmp(mode,'train')

    w = zeros(Nn,NN); %learned recurrent matrix
    W = zeros(Nout,NN); %output matrix
    PS = 1 * eye(NN); %inverse covariance matrix
    
elseif strcmp(mode, 'test')
    
    w = lrn.w; %learned recurrent matrix
    W = lrn.W; %output matrix
    
end

%% Initalize static parameters and network states

Jr = gr * J; %scaled recurrent of target-generating network
uFout = gfout * ufout; %scaled "output" input
uFin = gfin * ufin; %scaled input
uFhint = gfhint * ufhint; %scaled "hint" input

x_teach = 1e0 * randn(NN,1); %target-generating network state
x_learn = x_teach; %full-FORCE task-performing network state
z = zeros(Nn,1); %state vector of learned dyanamics

ttrial = inf; %time index, for calculating the end of a trial
TTrial = 0; %length of a trial, will change within the simulation loop
EZ = 0; %used to compute running update of Error variance
NF = 0; %same, but target variance

ttime = 0; %counts the number of trials completed so far, to terminate the simulation loop

%% Run simulation

while ttime <= TRLS+Tinit
    
    if ttrial > TTrial %new trial starts, generate new inputs and outputs
        
        ttrial = 1; %start counter for elapsed time in a trial
        
        switch task
            case 'ready_set_go'
                [fin,fout,fhint,TTrial] = trial_ready_set_go(p,hint);
                
        end
        
        %compute the inputs to each unit
        Fout = uFout * fout;
        Fin = uFin * fin;
        Fhint = uFhint* fhint;
        
        Zs = zeros(Nout,TTrial); %to accumulate generated outputs, for plotting
        
        ttime = ttime + 1; %increment trial counter by 1
        
    end
    
    %target generating network
    f = Jr * tanh(x_teach) + Fout(:,ttrial) + Fhint(:,ttrial);
    xinf = f + Fin(:,ttrial);
    x_teach = xinf + (x_teach - xinf) * etaux;
    f = V' * f;
    
    %full-FORCE task-performing network
    x_learn = x_learn + (dt/taux) * (-x_learn + ...
        (V * z + Fin(:,ttrial))) + sqrt(eta * dt) * randn(NN,1);
    r = tanh(x_learn);
    
    %compute the feedback from learning
    if ttime > Tinit
        Z = W * r;
        z = w * r;
    %if initialization period hasn't elapsed, used the target
    else
        z = f;
        Z = fout(:,ttrial);
    end
    
    %save the generated output for plotting
    Zs(:,ttrial) = Z;
    
    %do RLS
    if rand < 1/DTRLS && ttime > Tinit && strcmp(mode,'train')
        
        %fF
        xP = PS * r;
        k = (1 + r' * xP)\xP';
        
        PS = PS - xP * k;
        w = w - (z - f) * k;
        W = W - (Z - fout(:,ttrial)) * k;
        
    end
      
    % text output, plot things, and flush gathered data
    if ttrial == TTrial
             
        clc;
        fprintf('%s fullFORCE \nfullFORCE Error: %g\n%g trials of %g \n', ...
            mode, 100 * EZ/NF, ttime, (TRLS+Tinit));
        
        set(lh(1),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',Zs(1,:));
        set(lh(2),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',fhint(1,:));
        set(lh(3),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',fout(1,:));
        set(lh(4),'XData',p.dt*[ttrial-TTrial+1:ttrial],'YData',fin(1,:));
        
        drawnow;
        
    end
    
    %compute running value of normalized error
    if ttime > Tinit
        
        EZ = EZ + (Z - fout(:,ttrial))' * (Z - fout(:,ttrial));
        NF = NF + fout(:,ttrial)' * fout(:,ttrial);
        
    end
    
    ttrial = ttrial + 1; %increment trial time counter
    
end

if strcmp(mode,'train')
    
    %save learned matrices to structs
    lrn.w = w;
    lrn.W = W;
    lrn.PS = PS;
    
    varargout{1} = lrn;
    
elseif strcmp(mode,'test')
    
    %final error
    ERR(1)= 100 * EZ/NF;
    
    varargout{1} = ERR;
    
end

delete(lh); %remove old line handles

