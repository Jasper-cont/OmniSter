function graph_calibration_pose(objPoints1,rvecs1,tvecs1,objPoints2, ...
                                rvecs2, tvecs2, rvec, tvec, boardSize)
                            
fC_pose = zeros(3, length(objPoints1), length(rvecs1));

for frame = 1:length(rvecs1)
    rot = rodrigues(rvecs1(:,frame));
    for point = 1:length(objPoints1)
        corner = objPoints1(:,point,frame);
        fC_pose(:,point,frame) = rot * corner + tvecs1(:,frame);
    end
end

figure
hold on

for frame = 1:length(rvecs1)
    % Assuming chessboards
    outer_corner = zeros(3,4);
    outer_corner = [fC_pose(:,1,frame) fC_pose(:,boardSize(1),frame) ...
                    fC_pose(:,length(objPoints1),frame) ...
                    fC_pose(:,length(objPoints1) - boardSize(1) +1,frame)];
    
    fill3(outer_corner(1,:), outer_corner(2,:), outer_corner(3,:), 'b', 'facealpha', 0.1);
    plot3(outer_corner(1,1), outer_corner(2,1), outer_corner(3,1), 'k.');            
    %plot3(fC_pose(1,:,frame), fC_pose(2,:,frame), fC_pose(3,:,frame), 'k.');
end

grid on

plot3(get(gca,'XLim'),[0 0],[0 0],'r');
plot3([0 0],[0 0],get(gca,'ZLim'),'g');
plot3([0 0],get(gca,'YLim'),[0 0],'b');

hold off

end

