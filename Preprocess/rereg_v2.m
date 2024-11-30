% MATLAB Code for Image Registration and Overlapping

% Set up the directories
dm_folder = '../data/separate_images/Low_energy/';       % Folder containing DM images
cm_folder = '../data/separate_images/Subtracted/';       % Folder containing CM images
output_base_folder = './rereg_v2/';                      % Base folder to save the output images

% Create the output base folder if it doesn't exist
if ~exist(output_base_folder, 'dir')
    mkdir(output_base_folder);
end

% Read the annotations CSV with variable names preserved
annotations = readtable('../data/Radiology_hand_drawn_segmentations_v2.csv', 'PreserveVariableNames', true);

% Clean up the filenames in the annotations
annotations_filename = strrep(annotations.('#filename'), ' .jpg', '.jpg');
annotations_filename = strrep(annotations_filename, '.jpg', '');

% Create a map of annotations for quick lookup
annotations_map = containers.Map();
for i = 1:height(annotations)
    filename = annotations_filename{i};
    if ~isKey(annotations_map, filename)
        annotations_map(filename) = [];
    end
    annotations_map(filename) = [annotations_map(filename); annotations(i, :)];
end

% Get all DM and CM image filenames
dm_images = dir(fullfile(dm_folder, '*.jpg'));
cm_images = dir(fullfile(cm_folder, '*.jpg'));

% Combine DM and CM images
all_images = [dm_images; cm_images];

% Extract patient ID, side, type, view from the image names
all_image_names = {all_images.name}';
tokens = regexp(all_image_names, '(P\d+)_(L|R)_(DM|CM)_(CC|MLO)', 'tokens');

% Initialize arrays to store the extracted information
patient_ids = cell(size(tokens));
sides = cell(size(tokens));
types = cell(size(tokens));
views = cell(size(tokens));

for i = 1:length(tokens)
    if ~isempty(tokens{i})
        patient_ids{i} = tokens{i}{1}{1};
        sides{i} = tokens{i}{1}{2};
        types{i} = tokens{i}{1}{3};
        views{i} = tokens{i}{1}{4};
    else
        % Handle filenames that don't match the pattern
        patient_ids{i} = '';
        sides{i} = '';
        types{i} = '';
        views{i} = '';
    end
end

% Create the images_table including patient_id, side, type, view, filename
images_table = table(all_image_names, patient_ids, sides, types, views);
images_table.Properties.VariableNames = {'image_name', 'patient_id', 'side', 'type', 'view'};

% Remove entries with empty patient IDs (filenames that didn't match the pattern)
valid_entries = ~cellfun(@isempty, patient_ids);
images_table = images_table(valid_entries, :);

% Now, create a list of unique patient_id, side, view combinations
group_ids = strcat(images_table.patient_id, '_', images_table.side, '_', images_table.view);
[unique_groups, ~, idx_group] = unique(group_ids);

% Initialize an empty struct array to store image pairs
image_pairs = struct('fixed', {}, 'moving', {}, 'has_CM_annotation', {}, 'has_DM_annotation', {}, 'both_have_annotations', {});

for i = 1:length(unique_groups)
    group = unique_groups{i};
    group_indices = strcmp(group_ids, group);
    group_table = images_table(group_indices, :);

    % Check for CM and DM images
    has_CM = any(strcmp(group_table.type, 'CM'));
    has_DM = any(strcmp(group_table.type, 'DM'));

    % Proceed only if both images are available
    if ~has_CM || ~has_DM
        continue;
    end

    % Get the CM and DM image names
    cm_row = group_table(strcmp(group_table.type, 'CM'), :);
    dm_row = group_table(strcmp(group_table.type, 'DM'), :);

    % Extract patient_id, side, view
    patient_id = group_table.patient_id{1};
    side = group_table.side{1};
    view = group_table.view{1};

    % Check if annotations are available for CM and DM images
    cm_image_name = cm_row.image_name{1}(1:end-4);
    dm_image_name = dm_row.image_name{1}(1:end-4);

    has_CM_annotation = isKey(annotations_map, cm_image_name);
    has_DM_annotation = isKey(annotations_map, dm_image_name);

    % Decide which image to use as fixed and moving
    fixed_type = 'CM';
    moving_type = 'DM';
    fixed_image_name = cm_row.image_name{1};
    moving_image_name = dm_row.image_name{1};

    % Create the image_pair struct
    image_pair.fixed.patient_id = patient_id;
    image_pair.fixed.side = side;
    image_pair.fixed.view = view;
    image_pair.fixed.type = fixed_type;
    image_pair.fixed.image_name = fixed_image_name;

    image_pair.moving.patient_id = patient_id;
    image_pair.moving.side = side;
    image_pair.moving.view = view;
    image_pair.moving.type = moving_type;
    image_pair.moving.image_name = moving_image_name;

    image_pair.has_CM_annotation = has_CM_annotation;
    image_pair.has_DM_annotation = has_DM_annotation;
    image_pair.both_have_annotations = has_CM_annotation && has_DM_annotation;

    % Append to image_pairs array
    image_pairs(end+1) = image_pair;
end

% Now, process all image pairs
num_pairs = length(image_pairs);

for idx = 1:num_pairs
    pair = image_pairs(idx);

    fixed_image_name = pair.fixed.image_name;
    moving_image_name = pair.moving.image_name;

    % Paths
    fixed_image_path = fullfile(cm_folder, fixed_image_name);
    moving_image_path = fullfile(dm_folder, moving_image_name);

    % Check if images exist
    if ~exist(fixed_image_path, 'file') || ~exist(moving_image_path, 'file')
        fprintf('Image files not found for pair %d: %s, %s\n', idx, fixed_image_name, moving_image_name);
        continue;
    end

    % Create the output folder
    patient_folder = fullfile(output_base_folder, pair.fixed.patient_id);
    if ~exist(patient_folder, 'dir')
        mkdir(patient_folder);
    end
    subfolder_name = sprintf('%s_%s_%s', pair.fixed.patient_id, pair.fixed.side, pair.fixed.view);
    subfolder_path = fullfile(patient_folder, subfolder_name);
    if ~exist(subfolder_path, 'dir')
        mkdir(subfolder_path);
    end

    % Define output file names
    fixed_output_name = sprintf('%s_fixed.jpg', fixed_image_name(1:end-4));
    fixed_output_path = fullfile(subfolder_path, fixed_output_name);

    moving_output_name = sprintf('%s_moving.jpg', moving_image_name(1:end-4));
    moving_output_path = fullfile(subfolder_path, moving_output_name);

    registered_output_name = sprintf('%s_registered.jpg', moving_image_name(1:end-4));
    registered_output_path = fullfile(subfolder_path, registered_output_name);

    color_output_name = sprintf('%s_color.jpg', fixed_image_name(1:end-4));
    color_output_path = fullfile(subfolder_path, color_output_name);

    transformation_output_name = sprintf('transformation_matrix_%s.mat', fixed_image_name(1:end-4));
    transformation_output_path = fullfile(subfolder_path, transformation_output_name);

    % Check if processed images already exist
    if exist(fixed_output_path, 'file') && exist(moving_output_path, 'file') && ...
       exist(registered_output_path, 'file') && exist(color_output_path, 'file') && ...
       exist(transformation_output_path, 'file')
        fprintf('Processed images already exist for pair %d: %s and %s. Skipping...\n', idx, fixed_image_name, moving_image_name);
        continue;
    end

    % Load the images
    fixed = imread(fixed_image_path);
    moving = imread(moving_image_path);

    % Convert to grayscale if needed
    if size(fixed, 3) == 3
        fixed = rgb2gray(fixed);
    end
    if size(moving, 3) == 3
        moving = rgb2gray(moving);
    end

    % Display progress
    fprintf('Processing image pair %d/%d: %s and %s\n', idx, num_pairs, fixed_image_name, moving_image_name);

    % Set up the optimizer and metric for multimodal registration
    [optimizer, metric] = imregconfig('multimodal');

    % Adjust optimizer settings for better results
    optimizer.InitialRadius = optimizer.InitialRadius / 3.5;
    optimizer.MaximumIterations = 300;

    % Perform initial similarity transformation registration
    tformSimilarity = imregtform(moving, fixed, 'similarity', optimizer, metric);

    % Apply the transformation to the moving image
    Rfixed = imref2d(size(fixed));
    movingRegistered = imwarp(moving, tformSimilarity, 'OutputView', Rfixed);

    % Refine the registration using affine transformation with the initial condition
    tformAffine = imregtform(moving, fixed, 'affine', optimizer, metric, 'InitialTransformation', tformSimilarity);

    % Apply the affine transformation
    registered = imwarp(moving, tformAffine, 'OutputView', Rfixed);

    % Save images
    imwrite(fixed, fixed_output_path);
    imwrite(moving, moving_output_path);
    imwrite(registered, registered_output_path);

    % Generate the color image
    subtracted_image = fixed;
    low_energy_image = registered;

    color_image = zeros(size(fixed,1), size(fixed,2), 3, 'uint8');
    color_image(:,:,1) = low_energy_image; % R channel
    color_image(:,:,2) = subtracted_image; % G channel
    color_image(:,:,3) = subtracted_image; % B channel

    % Save the color image
    imwrite(color_image, color_output_path);

    % Save the transformation matrix
    T = tformAffine.T;
    save(transformation_output_path, 'T');

    % Handle annotations if present
    if pair.has_DM_annotation
        fprintf('Transforming annotations from DM to CM coordinate space.\n');
        dm_image_base = moving_image_name(1:end-4);
        dm_annotations_table = annotations_map(dm_image_base);
        mask_moving = createAnnotationMask(size(moving), dm_annotations_table);
        mask_registered = imwarp(mask_moving, tformAffine, 'OutputView', Rfixed, 'Interp', 'nearest');
        mask_output_name = sprintf('%s_registered_mask.png', moving_image_name(1:end-4));
        mask_output_path = fullfile(subfolder_path, mask_output_name);
        imwrite(mask_registered, mask_output_path);
    end

    if pair.has_CM_annotation
        fprintf('Annotations exist for CM image (fixed image).\n');
        % Optionally, save the mask for the CM image
        cm_image_base = fixed_image_name(1:end-4);
        cm_annotations_table = annotations_map(cm_image_base);
        mask_fixed = createAnnotationMask(size(fixed), cm_annotations_table);
        mask_output_name = sprintf('%s_fixed_mask.png', fixed_image_name(1:end-4));
        mask_output_path = fullfile(subfolder_path, mask_output_name);
        imwrite(mask_fixed, mask_output_path);
    end

    fprintf('Finished processing image pair %d/%d: %s and %s\n', idx, num_pairs, fixed_image_name, moving_image_name);
end

disp('All image pairs have been processed.');

% Function to create binary mask from annotations
function mask = createAnnotationMask(image_size, annotations_table)
    mask = false(image_size);
    for i = 1:height(annotations_table)
        ann_str = annotations_table{i, 'region_shape_attributes'}{1};
        ann_data = jsondecode(ann_str);
        shape_type = ann_data.name;
        if strcmp(shape_type, 'polygon') || strcmp(shape_type, 'polyline')
            x = ann_data.all_points_x;
            y = ann_data.all_points_y;
            mask = mask | poly2mask(x, y, image_size(1), image_size(2));
        elseif strcmp(shape_type, 'ellipse')
            [X, Y] = meshgrid(1:image_size(2), 1:image_size(1));
            cx = ann_data.cx;
            cy = ann_data.cy;
            rx = ann_data.rx;
            ry = ann_data.ry;
            ellipse_mask = ((X - cx)/rx).^2 + ((Y - cy)/ry).^2 <= 1;
            mask = mask | ellipse_mask;
        elseif strcmp(shape_type, 'circle')
            [X, Y] = meshgrid(1:image_size(2), 1:image_size(1));
            cx = ann_data.cx;
            cy = ann_data.cy;
            r = ann_data.r;
            circle_mask = (X - cx).^2 + (Y - cy).^2 <= r^2;
            mask = mask | circle_mask;
        elseif strcmp(shape_type, 'point')
            [X, Y] = meshgrid(1:image_size(2), 1:image_size(1));
            cx = ann_data.cx;
            cy = ann_data.cy;
            r = 5; % Small radius
            point_mask = (X - cx).^2 + (Y - cy).^2 <= r^2;
            mask = mask | point_mask;
        else
            fprintf('Unrecognized shape type: %s\n', shape_type);
            continue;
        end
    end
    mask = uint8(mask) * 255;
end
