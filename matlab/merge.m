function merge( img_path )

 img_list = dir([img_path '*.bmp']);
 n_imgs = length(img_list);
 total = imread([img_path img_list(1).name]);
 for i = 2:3
   total = total + imread([img_path img_list(i).name]);
 end
 
 image(total, 'CDataMapping','scaled');
end

