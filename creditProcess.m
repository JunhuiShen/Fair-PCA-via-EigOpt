# Default of Credit Card Clients dataset
function [M,A,B] = creditProcess()

data = readtable('default_degree.csv');
data = data{:,:};
data = data(:,2:end);
data = sortrows(data);

% Vector of sensitive attributes
sensitive = data(:,1);

% Define the sensitive attribute
normalized = (sensitive <= 1);

% Extract the sensitive attribute from the dataset
data = data(:,2:end);

[m,n] = size(data); % Number of attributes

% Centering and normalizing the dataset
data = (data - repmat(mean(data),m,1))./repmat(std(data),m,1);

% Data for sensitive data
data_sensitive = data(find(normalized),:);

% Data for nonsenstivie data
data_nonsensitive = data(find(~normalized),:);

% Centering and normalizing the sensitive attributes
data_sensitive = (data_sensitive - repmat(mean(data_sensitive),size(data_sensitive,1),1));
data_nonsensitive = (data_nonsensitive - repmat(mean(data_nonsensitive),size(data_nonsensitive,1),1));

A = data_sensitive;
B = data_nonsensitive;
M = [A;B];
end
