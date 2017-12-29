function main()
    % Question 1 
    spiralImage_sdm = CreateZeroLevelSetImage();
    
    % Question 2
    iteration_total = 5000;
    delta_t = 1;
    curve_evolution(delta_t,iteration_total, spiralImage_sdm);
end

function spiralImage_sdm = CreateZeroLevelSetImage()
    % read image
    spiralPicFileName = strcat('spiral','.jpg');
    spiralImage = imread(spiralPicFileName);
    paddedSpiralImage = padarray(spiralImage, [5 5], 255, 'both');
    % pad the image to create a larger surrounding region of white pixels
    [spiralImageRows, spiralImageCols, spiralImageChannels] = size(paddedSpiralImage);
    
    % get channel values
    spiralImage_channel1 = spiralImage(:, :, 1);
    %spiralImage_channel2 = spiralImage(:, :, 2);
    %spiralImage_channel3 = spiralImage(:, :, 3); 
    % just checking - all these channels have the same values
    %equal_1_2 = isequal(spiralImage_channel1, spiralImage_channel2);
    %equal_2_3= isequal(spiralImage_channel2, spiralImage_channel3);
    %equal_1_3 = isequal(spiralImage_channel1, spiralImage_channel3);
  
    % exterior(black) to boundary
    spiralImage_bw1 = imbinarize(spiralImage_channel1);
    spiralImage_d1 = bwdist(spiralImage_bw1);
    
    % interior(white) to boundary
    spiralImage_bw2 = imcomplement(spiralImage_bw1);
    spiralImage_d2 = -bwdist(spiralImage_bw2);
    
    % add two matrices to form the SDM(signed distance matrix)
    spiralImage_sdm = spiralImage_d1 + spiralImage_d2;
    
    %figure;
    %surf(spiralImage_sdm);
    %figure;
    %contour(spiralImage_sdm, [0,0]);
end

function curve_evolution(time_step, iteration_num, original_psi)
    videoName = 'spiral.avi';
    videoWriter = VideoWriter(videoName);
    open(videoWriter);

    psi = original_psi;
    for c = 1:iteration_num
        % 1st order derivatives
        psi_x = (padarray(diff(psi,1,2), [0,1],'symmetric','post') + padarray(diff(psi,1,2), [0,1],'symmetric','pre'))./2; 
        psi_y = (padarray(diff(psi,1,1), [1,0],'symmetric','post') + padarray(diff(psi,1,1), [1,0],'symmetric','pre'))./2;
        
        % 2nd order derivatives
        psi_xx = (padarray(diff(psi_x,1,2), [0,1],'symmetric','post') + padarray(diff(psi_x,1,2), [0,1],'symmetric','pre'))/2; 
        psi_yy = (padarray(diff(psi_y,1,1), [1,0],'symmetric','post') + padarray(diff(psi_y,1,1), [1,0],'symmetric','pre'))./2;
        psi_xy = (padarray(diff(psi_x,1,1), [1,0],'symmetric','post') + padarray(diff(psi_x,1,1), [1,0],'symmetric','pre'))./2;
        
        gradient_norm_squared = (psi_x.^2 + psi_y.^2 + 1e-15);
        delta_psi = ...
            (psi_xx.*psi_y.^2 - 2*(psi_x.*psi_y.*psi_xy) + psi_yy.*psi_x.^2)...
            ./gradient_norm_squared;
            
        % hint from disscusion board
        % avoid instability at boundary, so removing some updates around the boundary
        delta_psi(gradient_norm_squared<0.001) = 0;
        
        psi = time_step*delta_psi +  psi;
        contour(psi,[0,0]);
  
        movieFrame = getframe(gcf);
        if (mod(c,10) == 0)
            writeVideo(videoWriter,movieFrame);
            drawnow;
        end
    end
end
    



    