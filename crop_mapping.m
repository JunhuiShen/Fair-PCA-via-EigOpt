function [M, A, B] = crop_mapping()
% Load and preprocess the Crop Mapping dataset

% Load the dataset from the CSV file
data = readtable('crop.csv');
data = data{:,:};
data = sortrows(data);

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute: Corns (1 for corn, 0 for other crops)
normalized = (sensitive == 1);

% Extract the sensitive attribute from the dataset
data = data(:,2:end-1);

% Get the number of instances (m) and features (n)
[m,n] = size(data);

% Center and normalize the dataset
data = (data - repmat(mean(data), m, 1)) ./ repmat(std(data), m, 1);

% Data for corn
data_corn = data(find(normalized), :);

% Data for other crops
data_other_crops = data(find(~normalized), :);

% Center and normalize the sensitive attributes
data_corn = (data_corn - repmat(mean(data_corn), size(data_corn, 1), 1));
data_other_crops = (data_other_crops - repmat(mean(data_other_crops), size(data_other_crops, 1), 1));

% Assign the data to output variables
A = data_corn;
B = data_other_crops;
M = [A; B];

end