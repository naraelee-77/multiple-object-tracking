function successes = success_plot(positions, ground_truth, plot_title, fileToSave, show)
%PRECISION_PLOT
%   Calculates precision for a series of distance thresholds (percentage of
%   frames where the distance to the ground truth is within the threshold).
%   The results are shown in a new figure if SHOW is true.
%
%   Accepts positions and ground truth as Nx2 matrices (for N frames), and
%   a title string.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	
	max_threshold = 50;  %used for graphs in the paper
	
	
	successes = zeros(max_threshold, 1);
	
% 	if size(positions,2) ~= size(ground_truth,2)
% % 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
% 		
% 		%just ignore any extra frames, in either results or ground truth
% 		n = min(size(positions,2), size(ground_truth,2));
% 		positions(:,n+1:end,:) = [];
% 		ground_truth(:,n+1:end,:) = [];
% 	end
% 	
% 	%calculate distances to ground truth over all frames
% 
%     positions(:,:,5:6)=positions(:,:,3:4);
%     positions(:,:,3:4)=positions(:,:,1:2)+positions(:,:,3:4);
%     ground_truth(:,:,5:6)=ground_truth(:,:,3:4);
%     ground_truth(:,:,3:4)=ground_truth(:,:,1:2)+ground_truth(:,:,3:4);
%     
%     intersect(:,:,1:2)=max(positions(:,:,1:2), ground_truth(:,:,1:2));
%     intersect(:,:,3:4)=min(positions(:,:,3:4), ground_truth(:,:,3:4));
%     intersect(:,:,5:6)=intersect(:,:,3:4)-intersect(:,:,1:2);
%     
%     i_areas=intersect(:,:,5).*intersect(:,:,6);
%     i_areas(intersect(:,:,5)<0 | intersect(:,:,6)<0) = 0;
%     u_areas=positions(:,:,5).*positions(:,:,6) + ground_truth(:,:,5).*ground_truth(:,:,6) - i_areas;
%     
%     ious = i_areas/u_areas;
    
	if size(positions,1) ~= size(ground_truth,1)
% 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
		
		%just ignore any extra frames, in either results or ground truth
		n = min(size(positions,1), size(ground_truth,1));
		positions(n+1:end,:) = [];
		ground_truth(n+1:end,:) = [];
	end
	
	%calculate distances to ground truth over all frames

%     positions(:,5:6)=positions(:,3:4);
%     positions(:,3:4)=positions(:,1:2)+positions(:,3:4);
%     ground_truth(:,5:6)=ground_truth(:,3:4);
%     ground_truth(:,3:4)=ground_truth(:,1:2)+ground_truth(:,3:4);
%     
%     intersect(:,1:2)=max(positions(:,1:2), ground_truth(:,1:2));
%     intersect(:,3:4)=min(positions(:,3:4), ground_truth(:,3:4));
%     intersect(:,5:6)=intersect(:,3:4)-intersect(:,1:2);
%     
%     i_areas=intersect(:,5).*intersect(:,6);
%     i_areas(intersect(:,5)<0 | intersect(:,6)<0) = 0;
%     u_areas=positions(:,5).*positions(:,6) + ground_truth(:,5).*ground_truth(:,6) - i_areas;
%     
%     ious = i_areas/u_areas;

    ious = overlap_ratio(positions, ground_truth);
    
	%compute successes
	for p = 1:max_threshold
		successes(p) = nnz(ious >= p/max_threshold) / numel(ious);
	end
	
	%plot the precisions
	if show == 1
% 		figure('Number','off', 'Name',['Successes - ' title])
        figure('Name',['Successes - ' plot_title])
        title(plot_title)
		plot(successes, 'k-', 'LineWidth',2)
        ylim([0 1]) %, 'FontSize', 24)
%         xlim([0 1], 'FontSize', 24)
		xlabel('Overlap Threshold', 'FontSize', 24), ylabel('Success Rate', 'FontSize', 24)
        saveas(gcf, fileToSave);
	end
	
end

