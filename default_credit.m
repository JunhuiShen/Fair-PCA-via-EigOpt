function [M, A, B] = default_credit()
% Load and preprocess the Default of Credit Card Clients dataset

% Load the dataset from the CSV file
data = readtable('default_degree.csv');
data = data{:,:};
data = data(:,2:end);
data = sortrows(data);

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute (1 for low education level, 0 for higher education levels)
normalized = (sensitive <= 1);

% Extract the sensitive attribute from the dataset
data = data(:,2:end);

% Get the number of instances (m) and features (n)
[m,n] = size(data);

% Center and normalize the dataset
data = (data - repmat(mean(data), m, 1)) ./ repmat(std(data), m, 1);

% Data for sensitive group (low education level)
data_sensitive = data(find(normalized), :);

% Data for non-sensitive group (higher education levels)
data_nonsensitive = data(find(~normalized), :);

% Center and normalize the sensitive attributes
data_sensitive = (data_sensitive - repmat(mean(data_sensitive), size(data_sensitive, 1), 1));
data_nonsensitive = (data_nonsensitive - repmat(mean(data_nonsensitive), size(data_nonsensitive, 1), 1));

% Assign the data to output variables
A = data_sensitive;
B = data_nonsensitive;
M = [A; B];

end
