%%%%%%%%%%%%%%%%%% Importing Calibration Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
fC = xml2struct('../data/calibration_parameters/front_camera_params.xml');
bC = xml2struct('../data/calibration_parameters/back_camera_params.xml');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% Extracting Relevant Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fC_rmss = textscan(fC.opencv_storage.rmss.Text, '%f', 'delimiter', ' ');
bC_rmss = textscan(bC.opencv_storage.rmss.Text, '%f', 'delimiter', ' ');

fC_rvec = textscan(fC.opencv_storage.rvec.data.Text, '%f', 'delimiter', ' ');
bC_rvec = textscan(bC.opencv_storage.rvec.data.Text, '%f', 'delimiter', ' ');

fC_idx = textscan(fC.opencv_storage.idx.data.Text, '%d', 'delimiter', ' ');
bC_idx = textscan(bC.opencv_storage.idx.data.Text, '%d', 'delimiter', ' ');

fC_list = textscan(fC.opencv_storage.detect_list_1.Text, '%s', 'delimiter', ' ');
bC_list = textscan(bC.opencv_storage.detect_list_1.Text, '%s', 'delimiter', ' ');

fC_list = strrep(fC_list, '../data/calibration_images/top_front/tf_', '');
fC_list = strrep(fC_list, '.bmp', '');
fC_id = cell(1,numel(fC_idx));
for i = 1:length(fC_idx)
   fC_id(i) = fC_list(fC_idx(i)+1);
end

bC_list = strrep(bC_list, '../data/calibration_images/top_back/tb_', '');
bC_list = strrep(bC_list, '.bmp', '');
bC_id = cell(1,numel(bC_idx));
for i = 1:length(bC_idx)
   bC_id(i) = bC_list(bC_idx(i)+1);
end

fC_rvecs_ = textscan(fC.opencv_storage.rvecs.Text, '%f', 'delimiter', ' ');
fC_rvecs = zeros(3, numel(fC_rvecs_) / 3);
for i = 1:(numel(fC_rvecs_)/3)
   fC_rvecs(:,i) = fC_rvecs_(i:i+2); 
end

bC_rvecs_ = textscan(bC.opencv_storage.rvecs.Text, '%f', 'delimiter', ' ');
bC_rvecs = zeros(3, numel(bC_rvecs_) / 3);
for i = 1:(numel(bC_rvecs_)/3)
   bC_rvecs(:,i) = bC_rvecs_(i:i+2); 
end

fC_tvecs_ = textscan(fC.opencv_storage.tvecs.Text, '%f', 'delimiter', ' ');
fC_tvecs = zeros(3, numel(fC_tvecs_) / 3);
for i = 1:(numel(fC_tvecs_)/3)
   fC_tvecs(:,i) = fC_tvecs_(i:i+2); 
end

bC_tvecs_ = textscan(bC.opencv_storage.tvecs.Text, '%f', 'delimiter', ' ');
bC_tvecs = zeros(3, numel(bC_tvecs_) / 3);
for i = 1:(numel(bC_tvecs_)/3)
   bC_tvecs(:,i) = bC_tvecs_(i:i+2); 
end

fC_objPoints_ = textscan(fC.opencv_storage.object_points.data.Text, '%f', ...
                        'delimiter', ' ');
fC_objPoints_frames_ = textscan(fC.opencv_storage.object_points.rows.Text, '%d');
fC_objPoints_total_ = textscan(fC.opencv_storage.object_points.cols.Text, '%d');
fC_objPoints = zeros(3, fC_objPoints_total_, fC_objPoints_frames_);
for i = 1:fC_objPoints_frames_
   for j = 1:fC_objPoints_total_
       fC_objPoints(:,j,i) = fC_objPoints_(3*j-2:3*j);
   end
end

bC_objPoints_ = textscan(bC.opencv_storage.object_points.data.Text, '%f', ...
                        'delimiter', ' ');
bC_objPoints_frames_ = textscan(bC.opencv_storage.object_points.rows.Text, '%d');
bC_objPoints_total_ = textscan(bC.opencv_storage.object_points.cols.Text, '%d');
bC_objPoints = zeros(3, bC_objPoints_total_, bC_objPoints_frames_);
for i = 1:bC_objPoints_frames_
   for j = 1:bC_objPoints_total_
       bC_objPoints(:,j,i) = bC_objPoints_(3*j-2:3*j);
   end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

