function [fin,fout,fhint,TTrial,ITI] = trial_ready_set_go(p,hint)
%%

ITIb = 0.4; %min. ITI time
ITIl = 2.0; % %mean ITI time
ITI = 2 * round(0.5 * ((ITIb + exprnd(ITIl)) * (1/p.dt))); %compute ITI for that trial

delay = round(((10 + (210 - 10) * rand) * p.taux) * 1e3) * 1e-3; %pick random pulse interval

event = round([0.05, delay, 0.05, delay, 0.5] * (1/p.dt)); %timing of event sequence, for each trial
TTrial = sum(event) + ITI; %time of current trial

IOITI = zeros(1,ITI/2); %empty vector
I2ITI = -0.5 * ones(1,ITI/2); %inter-trial input

IO1 = ones(1,event(1)); %pulse inputs
IO2_1 = delay * linspace(0,1.0,event(2)+event(3)/2); %hint upwards
IO2_2 = delay * linspace(1.0,0,event(4)+event(3)/2); %hint downwards

%turn off hint signal
if strcmp(hint,'nohint')
    IO2_1 = 0 * IO2_1;
    IO2_2 = 0 * IO2_2;
end

%output
IO3 = -0.5 + 1.5 * betapdf(linspace(0,1,event(5)),4,4)/max(betapdf(linspace(0,1,event(5)),4,4));%target output

%construct inputs, hints and outputs from above signals

%task input
fin = [IOITI, IO1,zeros(1,sum(event(2:5))),IOITI;...
    IOITI, zeros(1,sum(event(1:2))),IO1,zeros(1,sum(event(4:5))),IOITI];

%desired output signals, and hint signals
fhint = [IOITI, zeros(1,sum(event(1))), IO2_1,IO2_2,zeros(1,sum(event(5))),IOITI];
fout = [I2ITI, -0.5 * ones(1,sum(event(1:4))), IO3, I2ITI];