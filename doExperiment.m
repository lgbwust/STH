function doExperiment(dataset, alg)
% doExperiment: Learning to Hash
% Dell Zhang
% Birkbeck, University of London

%% 
load(['testbed/',dataset]);
f_learn = eval(['@',alg,'_learn']);
f_compress = eval(['@',alg,'_compress']);

%%
codeLen = 4:4:64;
hammRadius = 0:3;
maxbits = codeLen(end);
tic;
[model, codeTrain] = f_learn(feaTrain, gndTrain, k, metric, kernel, maxbits);
timeTrain = toc;
tic;
codeTest = f_compress(feaTest, model, kernel, maxbits);
timeTest = toc;
disp([timeTrain, timeTest]);

%%
m = length(codeLen);
n = length(hammRadius);
trueP = zeros(m,n);
trueR = zeros(m,n);
cateP = zeros(m,n);
cateR = zeros(m,n);
cateA = zeros(m,n);
for i = 1:m
    nbits = codeLen(i);
    disp(nbits);
    cbTrain = compactbit(codeTrain(:,1:nbits));
    cbTest  = compactbit(codeTest(:,1:nbits));
    hammTrainTest  = hammingDist(cbTest,cbTrain)';
    for j = 1:n
        Ret = (hammTrainTest <= hammRadius(j)+0.00001);
        [trueP(i,j), trueR(i,j)] = evaluate_macro(trueTrainTest, Ret);
        [cateP(i,j), cateR(i,j)] = evaluate_macro(cateTrainTest, Ret);
        cateA(i,j) = evaluate_classification(gndTrain, gndTest, Ret);
    end
end
trueF1 = F1_measure(trueP, trueR);
cateF1 = F1_measure(cateP, cateR);

%%
clear feaTrain feaTest;
clear gndTrain gndTest;
clear trueTrainTest cateTrainTest;
clear hammTrainTest Ret;
save(['results/',dataset,'_',alg]);
clear;

end
