function [X,FVAL,REASON,OUTPUT,POPULATION,SCORES] = ex13_func
fitnessFunction = @ex13;
nvars = 2;
options = optimoptions(@ga,'PopInitRange',[-1 ; 3]);
options = optimoptions(options,'PopulationSize',10);
options = optimoptions(options,'MutationFcn',{@mutationgaussian 1 1});
options = optimoptions(options,'Display','off');
options = optimoptions(options,'PlotFcns',{@gaplotbestf,@gaplotbestindiv, @gaplotdistance});
[X,FVAL,REASON,OUTPUT,POPULATION,SCORES] = ga(fitnessFunction,nvars,options);
end
