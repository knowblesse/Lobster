function output = oddRatio(lnegP, lposP)

output = log(...
    exp(lposP).*(1-exp(lnegP)) ./ exp(lnegP).*(1-exp(lposP))...
    );

end