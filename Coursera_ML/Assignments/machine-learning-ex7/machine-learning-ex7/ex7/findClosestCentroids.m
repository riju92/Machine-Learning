function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1);
for i = 1 : m   %for every point in dataset X
   dist = zeros(K,1);    % Initialize a vector to store the distance between i & each centroid
   for k = 1 : K
        dist(k) = sum((X(i,:) - centroids(k,:)).^2);  % Store the distance between i and centroids(k) in dist(k)
   end
[value, index] = min(dist);  % Ignoring the actual distance, extract the index of the minimum dist.

idx(i) = index;  % Assign the index of closest centroid to idx(i)

end


% =============================================================

end

