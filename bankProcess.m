function [M, A_orig, B_orig] = bankProcess()

% Preprocess the Taiwanese Credit dataset and return the centered/normalized data matrix M (with al samples)
% and the samples of each sensitive group A and B. 

addpath data/bank

data = csvread('bank.csv');

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute: 1 for aged (>= 65 years) and 0 for <65 years
normalized = (sensitive >= 65);

% Extract the sensitive attribute from the dataset
data = data(:,2:end-1);

[m,n] = size(data); % Number of attributes

% Centering and normalizing the dataset
data = (data - repmat(mean(data),m,1))./repmat(std(data),m,1);

% Ddata for aged
data_aged = data(find(normalized),:);

% Data for younger
data_young = data(find(~normalized),:);

% Centering and normalizing the sensitive attributes
data_young = (data_young - repmat(mean(data_young),size(data_young,1),1));
data_aged = (data_aged - repmat(mean(data_aged),size(data_aged,1),1));

M = data;
B_orig = M(find(normalized),:);
A_orig = M(find(~normalized),:);
A = data_young;
B = data_aged;

end
