%% code for AE-RNN
for i = 1:length(animal)
    for j = 1:length(state)
        session = sess{i};
        for k = 1:length(session)
            dpath = [home,animal{i},'\',state{j},'\Sess',session{k}];

            if exist(dpath,'dir')==0
                continue
            else

                [elab,StartEnd] = func_alignBehTime(home,animal{i},state{j},session{k});

                Arr = load([dpath,'Res\','NeuTrace.mat']);
                NeuTraceMat = Arr.NeuTraceMat;
                disp([animal{i},' session ',session{k} ' has ',num2str(size(NeuTraceMat,1)), ' neurons'])

                Normalized = func_CalcDeltaf(NeuTraceMat,1,length(NeuTraceMat));
                NeuTraceMat = matsmooth(Normalized,10);

                [mtrial,ltrial] = func_getTrialData(NeuTraceMat,elab,0,60);
                if length(mtrial)<5
                    continue
                end

                combinedmat = [];
                rid = randsample(length(mtrial),length(mtrial));
                for ix = 1:length(rid)
                    autoenc = trainAutoencoder(mtrial{ix},8);
                    all_com{end+1} = autoenc.encode(mtrial{ix});
                    all_lab{end+1} = lab;
                end
            end
        end
    end
end
valaccu = [];
control = 0;
for iter = 1:200
    disp(iter)
    vid = [randsample(1:75,3),randsample(76:134,3),randsample(135:172,3)];
    tid = setdiff(1:length(all_com),vid);
    XTrain = all_com(tid);
    YTrain = [];
    for ixx = 1:length(tid)
        labtid = all_lab(tid);
        YTrain = [YTrain;categorical(labtid{ixx})];
    end
    XVal = all_com(vid);
    YVal = [];
    for ixx = 1:length(vid)
        labvid = all_lab(vid);
        YVal = [YVal;categorical(labvid{ixx})];
    end
    if control
        shuffleid = randperm(length(YTrain));
        YTrain = YTrain(shuffleid);
    end
    inputSize = 8;
    numHiddenUnits = 100;
    numClasses = 3;

    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
    maxEpochs = 200;
    miniBatchSize = 20;

    options = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu', ...
        'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'ValidationData',{XVal,YVal}, ...
        'ValidationFrequency',30, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'Verbose',0, ...
        'Plots','none');
    [net,info] = trainNetwork(XTrain,YTrain,layers,options);
    valaccu = [valaccu; info.ValidationAccuracy(end)];

    if max(info.ValidationAccuracy)>90
        save('.\bestmodel.mat','net') % Save the best model though useless 
    end
    netpred = predict(net,XVal);
    [~,netpredlab] = max(netpred,[],2);

    disp(mean(valaccu))

end
