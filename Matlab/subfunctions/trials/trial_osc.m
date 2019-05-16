function [fin,fout,TTrial] = trial_osc(p)
%%

event = [0.05,1.95] * (1/p.dt); %timing of event sequence, for each trial
TTrial = sum(event);

%freql = [1,3];
%freq = linspace(freql(1),freql(2),TTrial/2);
Tfull = TTrial*p.dt*100; % time in units of tau
t = 100*p.dt*(1:TTrial); % time in units of tau
freq =  t/Tfull.*(6*t/Tfull + 3);
freq(end/2+1:end) = -3 + t(end/2+1:end)/Tfull.*(15 -6*t(end/2+1:end)/Tfull);

IO1 = ones(1,event(1));
%IO2 = sin(2*pi*freq.*linspace(0,TTrial/2-1,TTrial/2)*p.dt);
%IO2 = [IO2,-1*fliplr(IO2)];
IO2 = sin(2*pi*freq);

fin = [IO1, zeros(1,sum(event(2)))];
fout = IO2;
