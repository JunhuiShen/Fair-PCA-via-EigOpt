function [M, A, B] = cropProcess()
% Preprocess the Crop mapping using fused optical-radar data set 
% and return the centered/normalized data matrix M (with all samples)
% and the samples of each sensitive group A and B. 
data = readtable('WinnipegDataset.csv');
data = data{:,:};
data = sortrows(data);

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute: Corns
normalized = (sensitive == 1);

% Extract the sensitive attribute from the dataset
data = data(:,2:end-1);

[m,n] = size(data); % Number of attributes

% Centering and normalizing the dataset
data = (data - repmat(mean(data),m,1))./repmat(std(data),m,1);

% Data for corn
data_corn = data(find(normalized),:);

% Data for other crops
data_other_crops = data(find(~normalized),:);

% Centering and normalizing the sensitive attributes
data_corn = (data_corn - repmat(mean(data_corn),size(data_corn,1),1));
data_other_crops = (data_other_crops - repmat(mean(data_other_crops),size(data_other_crops,1),1));

A = data_corn;
B = data_other_crops;
M = [A;B];
end