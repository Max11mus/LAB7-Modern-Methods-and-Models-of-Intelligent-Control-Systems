function [X,FVAL,REASON,OUTPUT,POPULATION,SCORES] = ex2_func
fitnessFunction = @ex2;
nvars = 1;
options = optimoptions(@ga,'PopInitRange',[-4 ; 1]);
options = optimoptions(options,'PopulationSize',10);
options = optimoptions(options,'MutationFcn',{@mutationgaussian 1 1});
options = optimoptions(options,'Display','off');
options = optimoptions(options,'PlotFcns',{@gaplotbestf, @gaplotbestindiv, @gaplotdistance});
[X,FVAL,REASON,OUTPUT,POPULATION,SCORES] = ga(fitnessFunction,nvars,options);
end
