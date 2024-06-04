function [M, A, B] = LFWProcess()
% Preprocess the LFW images. The output of the function is the centered
% images data as matrix M. Centered female images as group A and centered
% male images as group B.

% Add path to the images directory
addpath images

% Number of images and size of each image (flattened)
data_size = 13232;
img_size = 1764;

% Load the sex labels (0 for female, 1 for male)
fileID1 = fopen('sex.txt', 'r');
formatSpec = '%f';
sex = fscanf(fileID1, formatSpec);
fclose(fileID1);

% Initialize the images matrix
images = zeros(data_size, img_size);

% Load each image and store it in the images matrix
for ell = 0 : (data_size - 1)
    fileIDtmp = fopen(strcat('img', num2str(ell), '.txt'));
    formatSpec = '%f';
    array = fscanf(fileIDtmp, formatSpec);
    images(ell+1, :) = transpose(array);
    fclose(fileIDtmp);
end

% Normalize the images
images = images / 255;

% Center the images
mm = mean(images, 1);
images_centered = images - repmat(mm, data_size, 1);

% Separate and center female images
images_female = images(find(~sex), :);
female_mean = mean(images_female, 1);
size_female = size(images_female);
images_female_centered = images_female - repmat(female_mean, size_female(1), 1);

% Separate and center male images
images_male = images(find(sex), :);  
male_mean = mean(images_male, 1);
size_male = size(images_male);
images_male_centered = images_male - repmat(male_mean, size_male(1), 1);

% Assign the data to output variables
M = images_centered;
A = images_female_centered;
B = images_male_centered;

end