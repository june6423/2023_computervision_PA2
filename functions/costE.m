function [F]=costE(x, param)
    nimg = length(param.uv); % Number of camera poses.
    uv = param.uv;
    K = param.K;  
    
    % Extract R, T, X
    [Rvec,Tvec,X] = deserialize(x,nimg);
    nXn=0;
    for i=1:nimg
        nXn = nXn + length(uv{i}); end %number of reprojection errors
    
    F = zeros(2*nXn,1); 
    
    count = 1;
    for i = 1:nimg        
        % Rotation, Translation, [X, Y, Z]
        X_idx = uv{i}(4,:); nXi = size(X_idx, 2);
        R = RotationVector_to_RotationMatrix(Rvec(:,i)); T = Tvec(:,i); Xi = X(:,X_idx);   

        for j = 1:nXi
            proj = K * (R * Xi(:,j) + T);
            proj = proj / proj(3);
            F(count) = proj(1)-uv{i}(1,j);
            F(count+1) = proj(2)-uv{i}(2,j);
            count = count + 2;
        end
    end
end