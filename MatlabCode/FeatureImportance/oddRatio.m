function output = oddRatio(lposP, lnegP)
%% see Mladenic, D. & Grobelnik, M. (1999)
output = log(...
    exp(lposP).*(1-exp(lnegP)) ./ exp(lnegP).*(1-exp(lposP))...
    );

end