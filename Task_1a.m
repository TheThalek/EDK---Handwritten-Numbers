%% Getting data from supplied data-set
clear;
rawdata = load('datasets/data_all.mat');

%Content of data_all.mat: num_test, num_train, testlab, testv, trainlab, trainv,vec_size 
data.train_labels = rawdata.trainlab;
data.train_data = rawdata.trainv;
data.test_labels = rawdata.testlab;
data.test_data = rawdata.testv;

%% Variables
number_of_chunks = 60;
chunk_training_size = rawdata.num_train / number_of_chunks;
chunk_test_size = floor(rawdata.num_test / number_of_chunks);

num_of_classes = 10;


%% Finding matrices 
confusion_matrix = zeros(num_of_classes, num_of_classes);
classification_table = zeros(size(data.test_data, 1), 2);


for chunk_index = 1:number_of_chunks%number_of_chunks %Iteratin through all chunks
    %% Making a new data set for the i-chunk:
    %Finding data for training-set
    chunk_training_start = (chunk_index - 1)*chunk_training_size + 1; %Figuring out where the start of the chunk is
    training_interval = chunk_training_start:chunk_training_start + chunk_training_size - 1; %Making a vector containing the chunk data
    
    chunk_train_data = data.train_data(training_interval, :);
    chunk_train_labels = data.train_labels(training_interval, :);
    
    chunk_data.train_data = chunk_train_data;
    chunk_data.train_labels = chunk_train_labels;
    
    
    %Finding data for testing-set
    chunk_test_start = (chunk_index - 1)*chunk_test_size + 1;
    test_interval = chunk_test_start:(chunk_test_start + chunk_test_size -1);
    
    chunk_test_data = data.test_data(test_interval, :);
    chunk_test_labels = data.test_labels(test_interval, :);
    
    chunk_data.test_labels = chunk_test_labels;
    chunk_data.test_data = chunk_test_data;
    
    %% Finding the chunk confusion matrix and chunk classificaion table
    chunk_confusion_matrix = zeros(num_of_classes, num_of_classes);
    chunk_classification_table = zeros(size(chunk_data.test_data, 1), 2);
    distance_matrix = dist(chunk_data.train_data, chunk_data.test_data');

    %% Find the classification for each 
    for test_pred_index = 1:size(chunk_data.test_data)
         distance_vector = distance_matrix(:, test_pred_index);
         
         [min_value, min_index] = min(distance_vector);
         predicted_label = chunk_data.train_labels(min_index);
         actual_label = chunk_data.test_labels(test_pred_index);
         
         for i = 1:size(min_index) 
            chunk_confusion_matrix(actual_label+1, predicted_label+1) = chunk_confusion_matrix(actual_label+1, predicted_label+1) + 1; 
            chunk_classification_table(test_pred_index, 1) = predicted_label;
            chunk_classification_table(test_pred_index, 2) = actual_label;
         end
    end
    
    %% Adding the chunk matrices to get the final matrices
    confusion_matrix = confusion_matrix + chunk_confusion_matrix;
    chunk_interval = chunk_test_start:chunk_test_start + size(chunk_classification_table, 1)-1;
    classification_table(chunk_interval, :) = chunk_classification_table;
    
end


%% Displaying confusion matrix
classLabels = [0:9];
confusionchart(confusion_matrix, classLabels);

%% Plotting some misclassified images
for i = 1:size(classification_table)
    if(classification_table(i, 1) == 3 && classification_table(i, 2) == 3)
        x = zeros(28,28);
        x(:) = rawdata.testv(i,:);
        image(x);
        break
    end
end




