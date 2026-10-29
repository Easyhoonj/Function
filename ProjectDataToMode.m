function explained_energy_fraction_sum = ProjectDataToMode(ProjectedData, modes, lat)
    %% ProjectDataToMode - Projects data onto a set of modes.
    % written by: [Jin Juncheng] in 20261028.
    % Inputs:
    %   ProjectedData: The data to be projected(matrix), usually representing a trend field.
    %   modes: The modes onto which the data will be projected(matrix).
    %   lat: The latitude values corresponding to the data(vector).
    % Outputs:
    %   explained_energy_fraction: The fraction of energy explained by sum modes.
    %% First, we need to calculate the weights for each mode based on the latitude values.
    % simple deal with the weights for each mode based on the latitude values.
    % If the area of each grid cell is known, use area-based weights instead of cosd(lat)
    valid_index = isfinite(ProjectedData) & isfinite(modes(:,:,1));
    lat = lat(:)';
    [nLon, ~] = size(ProjectedData);
    weights = cosd(lat); 
    % Use the square root of the weights because least-squares coefficients are squared.
    weights = sqrt(repmat(weights, nLon, 1));
    nModes = size(modes, 3);
    num = nnz(valid_index);
    weighted_modes = zeros(num, nModes);
    for k = 1: nModes
        mode_k = modes(:, :, k);
        weighted_modes(:,k) = mode_k(valid_index).* weights(valid_index);
    end
    weighted_data = ProjectedData(valid_index).* weights(valid_index);
    % Check if the weighted modes are linearly independent
    if rank(weighted_modes) < nModes
        error('The modes are linearly dependent on the valid weighted grid.');
    end
    %% Second, we need to calculate the weighted modes by multiplying the modes with the weights.
    % Compute the Euclidean norms; their squares represent the energy carried by the field.
    data_norm = norm(weighted_data);
    % compute the ratio of the residual norm to the data norm, which represents the fraction of energy not explained by the modes.
    % unit is %
    % \ is symbol of Ordinary Least Squares in MATLAB.
    if data_norm == 0
        explained_energy_fraction = 100;
        return;
    end  
    explained_energy_fraction_sum = zeros(nModes,1);
    for k = 1:nModes
        prefix_coefficients = weighted_modes(:, 1:k)\ weighted_data;
        reconstructed = weighted_modes(:, 1:k)* prefix_coefficients;
        relative_residual = norm(weighted_data- reconstructed)/ data_norm;
        explained_energy_fraction_sum(k) = (1- relative_residual^2)*100; 
    end
end