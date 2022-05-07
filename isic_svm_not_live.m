%Load the directories of training and testing sets.
testDir = 'D:\datasets\isic2016\ISBI2016_ISIC_Part3_Test_Data'
trainDirBalanced = 'D:\datasets\isic2016\ISBI2016_ISIC_Part3_Training_Data'

TrainDir = 'D:\datasets\isic2016\unbalanced\ISBI2016_ISIC_Part3_Training_Data'
%Read the tables of both the train and testing ground truths, grabbing the image name and the corresponding diagnosis.

testGroundTruths = readtable("D:\datasets\isic2016\ISBI2016_ISIC_Part3_Test_GroundTruth.csv")
trainGroundTruthsBalanced = readtable("D:\datasets\isic2016\ISBI2016_ISIC_Part3_Training_GroundTruth.csv")

trainGroundTruths = readtable("D:\datasets\isic2016\unbalanced\ISBI2016_ISIC_Part3_Training_GroundTruth.csv")
%Load the training and testing images into a data store from their respective directories.

trainImdsBalanced = imageDatastore(trainDirBalanced, "IncludeSubfolders",true, "LabelSource","foldernames")
testImds = imageDatastore(testDir, "IncludeSubfolders",true, "LabelSource","foldernames")

trainBalanced = imageDatastore(TrainDir, "IncludeSubfolders",true,"LabelSource","foldernames")

%Grab a single image from the train data store, the specific one grabbed does not matter. Extract the hog features from the image and decide which is the optimal cell size to use, larger cell sizes are used when more large-scale spacial information is needed.
img = readimage(trainImdsBalanced, 20);
% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2])
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4])
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8])
[hog_16x16, vis16x16] = extractHOGFeatures(img, 'CellSize', [16 16])

% Show the image
figure; 
subplot(2,3,1:3); 
imshow(img);


% Visualize the HOG features

plot(vis2x2); 
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
 

plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});


%Define the cell size to be used during HOG feature extraction
cellSize = [8 8]
hogFeatureSize = length(hog_8x8)



 
%Loop over the training set and extract HOG features from each image . A similar procedure will be used to for the test set later on. This snippet deals with the balanced training data.

%get number of images in the image datastore
numImagesTrain = numel(trainImdsBalanced.Files)
trainingFeaturesBalanced = zeros(numImagesTrain, hogFeatureSize, 'single');
trainingLabelsBalanced = categorical(trainGroundTruthsBalanced{:,2})


for i = 1:numImagesTrain
    img = readimage(trainImdsBalanced, i);
    
   [rows, cols, numChannels] = size(img);
    img = imbinarize(img);
    
    trainingFeaturesBalanced(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.


%This snippet deals with the unbalanced training data.
%get number of images in the image datastore
numImagesTrain = numel(trainImds.Files)
trainingFeatures = zeros(numImagesTrain, hogFeatureSize, 'single');
trainingLabels = categorical(trainGroundTruths{:,2})


for i = 1:numImagesTrain
    img = readimage(trainImds, i);
    
   [rows, cols, numChannels] = size(img);
%     if (numChannels > 1)
%         img = rgb2gray(img);
%     end
    
% %     img = imresize(img, [209, 191] )
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

countcats(trainingLabels)


%Create an SVM template which grants greater flexibility in controlling the parameters of the model. Predictor data is standardised meaning each column is centred and scaled by the weighted column mean and standard deviation. 
%The classifier is then trained using the fit method, passing in the training data and the template SVM. Hyperparameter optimisation attempts to minimise the cross-validation loss by varying the parameters, this considerably adds to compute time. 
t = templateSVM('Standardize', true, 'KernelFunction', 'rbf')

classifierLinear = fitclinear(trainingFeaturesBalanced, trainingLabelsBalanced)
classifierSVM = fitcsvm(trainingFeaturesBalanced, trainingLabelsBalanced)

classifierSVMUnbalanced = fitcsvm(trainingFeatures, trainingLabels)

%Similar to the above procedure whereby HOG features are extracted from the test image set. 
numImagesTest = numel(testImds.Files)
testingFeatures = zeros(numImagesTest, hogFeatureSize, 'single');
testLabels = categorical(testGroundTruths{2:1:end,2});

for i = 1:numImagesTest
    img = readimage(testImds, i);
    
   [rows, cols, numChannels] = size(img);
%     if (numChannels > 1)
%         img = rgb2gray(img);
%     end
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    testingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end



%Metrics for the standard linear SVM
predictionsLinear = predict(classifierLinear, testingFeatures);
[cLinear,orderLinear] = confusionmat(testLabels, predictionsLinear);
chartLinear = confusionchart(cLinear, orderLinear)
[senLinear, specLinear, accLinear, f1Linear, pLinear] = calcMetrics(cLinear)
%Metrics for the standard non linear SVM

predictionsSVM = predict(classifierSVM, testingFeatures);
[cSVM,orderSVM] = confusionmat(testLabels, predictionsSVM);
chartSVM = confusionchart(cSVM, orderSVM)
[senSVM, specSVM, accSVM, f1SVM, pSVM] = calcMetrics(cSVM)

%Metrics for the unbalanced varient of the standard SVM classifier
predictionsSVMUnbalanced = predict(classifierSVMUnbalanced, testingFeatures);
[cSVMUnbalanced,orderSVMUnbalanced] = confusionmat(testLabels, predictionsSVMUnbalanced);
chartSVM = confusionchart(cSVMUnbalanced, orderSVMUnbalanced)
[senSVMUnbalanced, specSVMUnbalanced, accSVMUnbalanced, f1SVMUnbalanced, pSVMUnbalanced] = calcMetrics(cSVMUnbalanced)

function [sensitivity, specificity, accuracy, f1, p] = calcMetrics(c) 

    p = sum(diag(c)) / sum(c(1:1:end));
    

    sensitivity = c(1,1) / (c(1,1) + c(1,2));
    specificity = c(2,2) / (c(2,2) + c(2,1));
    accuracy = (c(1,1) + c(2,2)) /(c(1,1) + c(2,2) + c(2,1) + c(1,2));
    % jaccard_similiarity_index = c(1,1) / (c(1,1) + c(1,2) + c(2,1))
    % dice_similarity_coefficient = 2 * c(1,1) / (2 * c(1,1) + c(1,2) + c(2,1))
    f1 = c(1,1) / c(1,1) + 0.5 * (c(2,1) + c(1,2));
end

function v = visualise_svm(m, examples, labels)

    % Hint: for any model, m, trained with fitcsvm(),
    % you can access the data used to train it with:
    m.X
    m.Y
    
    % Add as many lines of code as you need below:
    xmin = min(m.X{:,1});
    ymin = min(m.X{:,2});
    
    xmax = max(m.X{:,1});
    ymax = max(m.X{:,2});
    
    grid = [];
    counter =1;
    for i = xmin : 0.10 : xmax
         for j = ymin : 0.10 : ymax
        %the value of the outer loop (x) is stored in the first column
        grid(counter,1) = i;
        %inner loop(y) is stored in the second column
        grid(counter,2) = j;
        %increment to make sure the row numbers begin at 1 and go from
        %there
        counter = counter + 1;
    end
    %incremment the counter for the x axis 
    counter = counter + 1;
    end
    predictions = predict(m, grid);
  %retain the plot points
    hold on
    %plot the linear separator line using the grid generated previously 
    gscatter(grid(:,1), grid(:,2), predictions)
    hold off
    %superimose the original data ontop of the 
    hold on
    v = gscatter(examples{:,1}, examples{:,2}, labels);
    hold off
end