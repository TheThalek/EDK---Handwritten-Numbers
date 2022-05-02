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

%% Making M clusters
for i = 1:num_of_classes
   if()
       
   end
end

[idx_1, C_1] = kmeans(data.train_data, M);



%% 


 