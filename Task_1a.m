%% Getting data from supplied data-set
clear;
rawdata = load('datasets/data_all.mat');

%Content of data_all.mat: num_test, num_train, testlab, testv, trainlab, trainv,vec_size 
data.train_labels = rawdata.trainlab;
data.train_data = rawdata.trainv;
data.test_labels = rawdata.testlab;
data.test_data = rawdata.testv;

%% Variables
chunk_training_size = 6000;
chunk_test_size = 1000;

training_chunks = (size(data.train_labels) / chunk_training_size);
N_training_chunks = training_chunks(1);

test_chunks = (size(data.test_labels) / chunk_test_size);
N_test_chunks = test_chunks(1);

num_of_classes = 10;


%% Finding matrices 
confusion_matrix = zeros(num_of_classes, num_of_classes);
classification_table = zeros(size(data.test_data, 1), 2);


for chunk_index = 1:N_training_chunks %Iteratin through all chunks
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
    test_interval = chunk_test_start:chunk_test_start + chunk_test_size -1;
    
    chunk_test_data = data.test_data(test_interval, :);
    chunk_test_labels = data.test_labels(test_interval, :);
    
    chunk_data.test_labels = chunk_test_labels;
    chunk_data.test_data = chunk_test_data;
    
    %% Finding the chunk confusion matrix and chunk classificaion table
    chunk_confusion_matrix = zeros(num_of_classes, num_of_classes);
    chunk_classification_table = zeros(size(chunk_data.test_data, 1), 2);
    distance_matrix = dist(chunk_data.test_data, chunk_data.train_data');
    
    k = 1;
    [min_values, min_indices] = mink(distance_matrix', k, 1);
    
    size_chunk_test_data = size(chunk_data.test_data, 1);

    for j = 1:size_chunk_test_data
        min_index = min_indices(:, j);
        classified_labels = chunk_data.train_labels(min_index);
        disp(classified_labels);
        
        unique_classes = unique(classified_labels);
        disp(unique_classes);
        bins = histcounts(classified_labels, unique_classes);
        modes = unique_classes(bins == max(bins));
        
        min_distance = -1;
        classified_label = -1;
        
        for k = 1:size(modes, 2)
            class = modes(1, k);
            distance = 0;
            
            for index = 1:size(min_index, 2)
                if class == tempate_labels(index)
                    distance = distance + min_values(index);
                end
            end
            
            if distance < min_distance || min_distance == -1
                min_distance = distance;
                
                classified_label = class + 1;
            end
    
        end
        
        correct_label = chunk_data.test_labels(j) + 1;
        chunk_confusion_matrix(correct_label, classified_label) = confusion_matrix(correct_label, classified_label) + 1;
       
        chunk_classification_table(j, :) = [correct_label - 1 classified_label -1];
    end
    
    %% Adding the chunk matrices to get the final matrices
    chunk_interval = chunk_test_start:chunk_test_start + size(chunk_classification_table, 1)-1;
    
    confusion_matrix = confusion_matrix + chunk_confusion_matrix; 
    classification_table(chunk_interval, :) = chunk_classification_table;
    
end

%% Plotting the matrices
%Displaying confusion matrix
%disp(confusion_matrix);
classLabels = [0:9];
%confusionchart(confusion_matrix, classLabels);



