function main()
    delta_t = 0.01;
    iteration_total = 2000;
    n = 6;
    sigma = 3;
    beta_0 = -10;
    beta_1 = 1;

    beetle_pic_filename = strcat('beetle','.jpg');
    beetle_color_image = imread(beetle_pic_filename);
    beetle_gray_image = rgb2gray(beetle_color_image); 
    [beetle_rows, beetle_cols, beetle_channels] = size(beetle_gray_image);
    
    % compute Phi
    beetle_gaussian_blurred = imgaussfilt(beetle_gray_image,sigma);
    [Ix, Iy] = gradient(double(beetle_gaussian_blurred));
    beetle_gaussian_norm = Ix.^2 + Iy.^2;
    phi = 1./(beetle_gaussian_norm.^n+1);
    
    % visualize stopping term, intensity is in [0,1] interval,
    % 0(black) -> 1(white)
    %figure;
    %imshow(phi);
    % this looks correct, since we want this stopping term to be small at
    % edges
    bettle_image_starting_psi = GetStartingPsi(beetle_rows, beetle_cols);

    segmentation_flow(delta_t,iteration_total, phi, beta_0, beta_1, bettle_image_starting_psi);

end

function bettle_image_starting_psi = GetStartingPsi(beetle_rows, beetle_cols)
    mask = zeros(beetle_rows, beetle_cols);
    mask(5:beetle_rows-4,5:beetle_cols-4) = 1;
    
    beetle_image_bw1 = mask;
    bettle_image_d1 = -bwdist(beetle_image_bw1);
    
    % only want to detect edges on the edge of beetle, not inside
    beetle_image_bw2 = ~mask;
    bettle_image_d2 = 0;
    %bettle_image_d2 = bwdist(beetle_image_bw2);
    
    bettle_image_starting_psi = bettle_image_d1 + bettle_image_d2;
    
    % to visualize the level set for the starting psi
    %figure;
    %surf(bettle_image_starting_psi);
end

function segmentation_flow(delta_t, iteration_total, phi, beta_0, beta_1, starting_psi)
    % for visualization with curve evolution
    beetle_pic_filename = strcat('beetle','.jpg');
    beetle_color_image = imread(beetle_pic_filename);
    
    % for video capture of curve evolution
    videoName = 'beetle.avi';
    videoWriter = VideoWriter(videoName);
    open(videoWriter);

    psi = starting_psi;
    
    for c = 1:iteration_total
        % region 1 - curvture term
        % 1st order derivatives
        psi_x = (padarray(diff(psi,1,2), [0,1],'symmetric','post') + padarray(diff(psi,1,2), [0,1],'symmetric','pre'))./2; 
        psi_y = (padarray(diff(psi,1,1), [1,0],'symmetric','post') + padarray(diff(psi,1,1), [1,0],'symmetric','pre'))./2;
        
        % 2nd order derivatives
        psi_xx = (padarray(diff(psi_x,1,2), [0,1],'symmetric','post') + padarray(diff(psi_x,1,2), [0,1],'symmetric','pre'))/2; 
        psi_yy = (padarray(diff(psi_y,1,1), [1,0],'symmetric','post') + padarray(diff(psi_y,1,1), [1,0],'symmetric','pre'))./2;
        psi_xy = (padarray(diff(psi_x,1,1), [1,0],'symmetric','post') + padarray(diff(psi_x,1,1), [1,0],'symmetric','pre'))./2;
        
        % curvature term
        gradient_norm_squared = (psi_x.^2 + psi_y.^2+1e-15);      
        delta_psi_curvature = (psi_xx.*psi_y.^2 - 2*(psi_x.*psi_y.*psi_xy) + psi_yy.*psi_x.^2)./gradient_norm_squared;
        delta_psi_curvature(gradient_norm_squared<0.001) = 0;
        
        % region 2 - constant term
        % forward and backward differences
        forward_diff_x = padarray(diff(psi,1,2), [0,1],0, 'post');
        backward_diff_x = padarray(diff(psi,1,2), [0,1],0, 'pre');
        forward_diff_y = padarray(diff(psi,1,1), [1,0],0,'post');
        backward_diff_y = padarray(diff(psi,1,1), [1,0],0,'pre');

        forward_diff_x_max = max(forward_diff_x, 0);
        forward_diff_x_min = min(forward_diff_x, 0);
        backward_diff_x_max = max(backward_diff_x, 0);
        backward_diff_x_min = min(backward_diff_x, 0);
        forward_diff_y_max = max(forward_diff_y, 0);
        forward_diff_y_min = min(forward_diff_y, 0);
        backward_diff_y_max = max(backward_diff_y, 0);
        backward_diff_y_min = min(backward_diff_y, 0);
        nabla_plus = (forward_diff_x_max.^2 + backward_diff_x_min.^2 +...
            forward_diff_y_max.^2 + backward_diff_y_min.^2).^(1/2);
        nabla_minus = (backward_diff_x_max.^2 + forward_diff_x_min.^2 + ...
            backward_diff_y_max.^2 + forward_diff_y_min.^2).^(1/2);
        
        % constant term
        delta_psi_constant = max(beta_0,0)*nabla_plus + min(beta_0,0)*nabla_minus;
        
        % region 3 - for debugging
        % this piece here was for conditional breakpoint to detect 'inf'
        % and 'NaN'
        %delta_psi_curvature_nan = sum(isnan(delta_psi_curvature) == 1);
        %delta_psi_curvature_nan_count = sum(delta_psi_curvature_nan);
        
        %delta_psi_curvature_inf = sum(isinf(delta_psi_curvature) == 1);
        %delta_psi_curvature_inf_count = sum(delta_psi_curvature_inf);
        
        %delta_psi_constant_nan = sum(isnan(delta_psi_constant) == 1);
        %delta_psi_constant_nan_count = sum(delta_psi_constant_nan);
        
        %delta_psi_constant_inf = sum(isinf(delta_psi_constant) == 1);
        %delta_psi_constant_inf_count = sum(delta_psi_constant_inf);
        
        % update Psi
        psi = delta_t*phi.*(beta_1*delta_psi_curvature + delta_psi_constant) + psi;
        
        % visualization
        image(beetle_color_image);
        hold on;
        contour(psi,[0,0],'LineColor','r');
        
        % write to avi every 5 steps
        movieFrame = getframe(gcf);
        if (c == 1 || mod(c,5) == 0)
            writeVideo(videoWriter,movieFrame);
            drawnow;
        end
    end
end




    