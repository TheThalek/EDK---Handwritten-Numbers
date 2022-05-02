%% Loading the data
clear;
rawdata = load('datasets/data_all.mat');

%Content of data_all.mat: num_test, num_train, testlab, testv, trainlab, trainv,vec_size 
data.train_labels = rawdata.trainlab;
data.train_data = rawdata.trainv;
data.test_labels = rawdata.testlab;
data.test_data = rawdata.testv;

%% Initializing the value of k and other variables
k = 7;
num_of_classes = 10;
M_num_of_clusters = 64;

confusion_matrix = zeros(num_of_classes, num_of_classes);
classification_table = zeros(size(data.test_data, 1), 2);

%% Making M clusters, 2a)
cluster_matrix = zeros(num_of_classes*M_num_of_clusters, size(data.train_data,2));

for i = 0:num_of_classes-1
    kmeans_training_data = zeros(size(data.train_data,1)/num_of_classes, size(data.train_data,2));
    k=1;
    for j = 1:size(data.train_labels, 1)
        if(data.train_labels(j) == i)
            
            kmeans_training_data(k, :) = data.train_data(j,:);
            k = k + 1;
        end
    end
    
    [idx_i , C_i] = kmeans(kmeans_training_data, M_num_of_clusters);
    cluster_matrix((i*64+1):((i+1)*64), :) = C_i; %Placing C_i into common matrix
    
end

%% Finding confusion matrix for NN, 2b)
distance_matrix = dist(cluster_matrix, data.test_data');

for test_pred_index = 1:size(data.test_data)
     distance_vector = distance_matrix(:, test_pred_index);

     [min_value, min_index] = min(distance_vector);
     predicted_label = floor((min_index-1)/64);
     actual_label = data.test_labels(test_pred_index);
     
     confusion_matrix(actual_label+1, predicted_label+1) = confusion_matrix(actual_label+1, predicted_label+1) + 1;
     classification_table(test_pred_index, 1) = predicted_label;
     classification_table(test_pred_index, 2) = actual_label;
     
end

classLabels = [0:9];
confusionchart(confusion_matrix, classLabels);








