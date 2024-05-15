# Bank Marketing dataset
function [M, A, B] = bankProcess()

addpath data/bank

data = csvread('bank.csv');

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute: 1 for aged (>= 65 years) and 0 for <65 years
normalized = (sensitive >= 65);

% Extract the sensitive attribute from the dataset
data = data(:,2:end);

[m,n] = size(data); % Number of attributes

% Centering and normalizing the dataset
data = (data - repmat(mean(data),m,1))./repmat(std(data),m,1);

% Ddata for sensitive group
data_sensitive = data(find(normalized),:);

% Data for non-sensitive group
data_nonsensitive = data(find(~normalized),:);

% Centering and normalizing the sensitive attributes
data_nonsensitive = (data_nonsensitive - repmat(mean(data_nonsensitive),size(data_nonsensitive,1),1));
data_sensitive = (data_sensitive - repmat(mean(data_sensitive),size(data_sensitive,1),1));

A = data_nonsensitive;
B = data_sensitive;
M = [A;B];

end
