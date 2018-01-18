function [fin,fout,TTrial] = trial_osc(p)
%%

event = [0.05,1.95] * (1/p.dt); %timing of event sequence, for each trial
TTrial = sum(event);

freql = [1,3];
freq = linspace(freql(1),freql(2),TTrial/2);

IO1 = ones(1,event(1));
IO2 = sin(2*pi*freq.*linspace(0,TTrial/2-1,TTrial/2)*p.dt);
IO2 = [IO2,-1*fliplr(IO2)];

fin = [IO1, zeros(1,sum(event(2)))];
fout = IO2;