function [Y, mapping] = LPP(X, w, sG, options)
% LPP function adapted for direct input Y (preprocessed data from previous layer).
% Default values
if nargin < 3, options = []; end

[nSmp, nFea] = size(X);
if size(W, 1) ~= nSmp, error('Dimension mismatch between similarity matrix W and data Y.'); end

% Normalize weight matrices
mG = max(max(abs(w)));
msG = max(abs(sG));  
maxNorm = max([mG msG]);
G = w./ maxNorm;
sG = sG ./ maxNorm;

% Balance graphs if required (assuming balance is set internally as needed)
balance = 1; % This should be set based on your requirements or calculations if different from 1
w_balanced = G + balance * sG;

% Compute diagonal matrix for normalization
D = full(sum(w_balanced, 2));

% Regularization and eigenvector computation
if ~isfield(options, 'Regu') || ~options.Regu
    [~, eigvector, eigvalue] = svds(w_balanced, min(nSmp, nFea));
else
    D = spdiags(sqrt(D), 0, nSmp, nSmp);
    [~, eigvector, eigvalue] = LGE(w_balanced, D, options, Y); % Passing Y if LGE expects it
end

% Remove small eigenvalues and corresponding eigenvectors
tol = 1e-3;
[eigvalue, eigIdx] = sort(eigvalue, 'ascend');
eigvalue(eigvalue < tol) = [];
eigvector(:, eigIdx(1:end-tol)) = [];

% Mapping and output
Y = Y * eigvector;
mapping.M = eigvector;
mapping.nSmp = nSmp;
mapping.eigvalue = eigvalue;

% If LGE indeed uses or requires information like DToPowerHalf or other preprocessing steps from Y,
% ensure these are handled appropriately based on the actual implementation of LGE and your pipeline.
