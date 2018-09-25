function precisions = precision_plot(positions, ground_truth, plot_title, fileToSave, show)
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

%     positions = positions(:,:, [2,1]) + positions(:,:, [4,3]) / 2;
%     ground_truth = ground_truth(:,:, [2,1]) + ground_truth(:,:, [4,3]) / 2;
% 	
% 	max_threshold = 50;  %used for graphs in the paper
% 	
% 	
% 	precisions = zeros(max_threshold, 1);
% 	
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
% 	distances = sqrt((positions(:,:,1) - ground_truth(:,:,1)).^2 + ...
% 				 	 (positions(:,:,2) - ground_truth(:,:,2)).^2);
% 	distances(isnan(distances)) = [];
% 
% 	%compute precisions
% 	for p = 1:max_threshold
% 		precisions(p) = nnz(distances <= p) / numel(distances);
% 	end
% 	
% 	%plot the precisions
% 	if show == 1
% % 		figure('Number','off', 'Name',['Precisions - ' title])
%         figure('Name',['Precisions - ' title])
% 		plot(precisions, 'k-', 'LineWidth',2)
%         ylim([0 1])
% 		xlabel('Threshold'), ylabel('Precision')
%         saveas(gcf, fileToSave);
%     end
    
    positions = positions(:, [2,1]) + positions(:, [4,3]) / 2;
    ground_truth = ground_truth(:, [2,1]) + ground_truth(:, [4,3]) / 2;
	
	max_threshold = 50;  %used for graphs in the paper
	
	
	precisions = zeros(max_threshold, 1);
	
	if size(positions,1) ~= size(ground_truth,1)
% 		fprintf('%12s - Number of ground truth frames does not match number of tracked frames.\n', title)
		
		%just ignore any extra frames, in either results or ground truth
		n = min(size(positions,1), size(ground_truth,1));
		positions(n+1:end,:) = [];
		ground_truth(n+1:end,:) = [];
	end
	
	%calculate distances to ground truth over all frames
	distances = sqrt((positions(:,1) - ground_truth(:,1)).^2 + ...
				 	 (positions(:,2) - ground_truth(:,2)).^2);
	distances(isnan(distances)) = [];

	%compute precisions
	for p = 1:max_threshold
		precisions(p) = nnz(distances <= p) / numel(distances);
	end
	
	%plot the precisions
	if show == 1
% 		figure('Number','off', 'Name',['Precisions - ' title])
        figure('Name',['Precisions - ' plot_title])
        title(plot_title, 'FontSize', 24)
		plot(precisions, 'k-', 'LineWidth',2)
%         xlim([0 max_threshold], 'FontSize', 24)
        ylim([0 1]) %, 'FontSize', 24)
		xlabel('Location Error Threshold'), ylabel('Precision')
        saveas(gcf, fileToSave);
	end
	
end

