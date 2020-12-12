%%%%%%%%%%%%%%%%% Plotting RMS vs image ID %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function graph_frame_rms(rmss, id)
    bar(rmss);
    set(gca, 'xticklabel',id, 'XTick', 1:numel(id));
    xlabel('Image ID');
    ylabel('Reprojection Error (px)');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%