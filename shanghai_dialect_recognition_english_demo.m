function shanghai_dialect_recognition_english_demo()
    % 初始化参数
    params = init_parameters();
    
    % 加载现有模型（如果存在）
    params = load_existing_model(params);
    
    while true
        fprintf('\n=== Shanghai Dialect Detection System ===\n');
        fprintf('1. Detect Audio File\n');
        fprintf('2. Train Model\n');
        fprintf('3. Quick Training (10+10 samples)\n');
        fprintf('4. Update Detection Parameters\n');
        fprintf('5. Display Current Parameters\n');
        fprintf('6. Exit\n');
        
        choice = input('Please select (1-6): ', 's');
        
        switch choice
            case '1'
                detect_audio_file(params);
            case '2'
                train_model(params);
            case '3'
                params = quick_train_model(params);  % 新增快速训练功能
            case '4'
                params = update_detection_parameters(params);  % 新增更新参数功能
            case '5'
                display_current_parameters(params);  % 新增显示参数功能
            case '6'
                fprintf('Exiting system\n');
                break;
            otherwise
                fprintf('Invalid selection\n');
        end
    end
end
%%
function params = init_parameters()
    % 初始化参数
    params.fs = 16000;           % 采样率
    params.frame_len = 0.025;    % 帧长25ms
    params.hop_len = 0.010;      % 帧移10ms
    params.min_duration = 1.0;   % 最小音频时长1秒
    
    % 沪语特征阈值（基于论文中的声学特征）
    params.shanghai_threshold = 0.6;      % 沪语判断阈值
    params.low_freq_weight = 0.241;         % 低频能量权重（调整）
    params.pitch_var_weight = 0.092;        % 基频变化权重（调整）
    params.zcr_weight = 0.580;              % 过零率权重（调整）
    params.energy_var_weight = 0.087;       % 能量变化权重（调整）
    
    % 模型训练参数
    params.model_file = 'shanghai_model.mat';  % 模型保存文件
    params.quick_train_file = 'shanghai_quick_model.mat';  % 快速训练模型文件
    
    % 训练统计信息
    params.training_stats = struct();
    params.training_stats.last_trained = 'Never trained';
    params.training_stats.samples_used = 0;
    params.training_stats.accuracy = 0;
end
%%
function params = load_existing_model(params)
    % 加载现有模型
    if exist(params.model_file, 'file')
        try
            loaded_data = load(params.model_file);
            if isfield(loaded_data, 'model') && isfield(loaded_data, 'params')
                fprintf('Found saved model, loading...\n');
                loaded_params = loaded_data.params;
                
                % 合并参数，保留新版本的新增字段
                params = merge_parameters(params, loaded_params);
                
                params.training_stats.last_trained = 'Loaded from saved file';
                fprintf('Model loaded successfully!\n');
            end
        catch ME
            fprintf('Failed to load model: %s\n', ME.message);
        end
    end
    
    % 加载快速训练模型（只有在字段存在时才执行）
    if isfield(params, 'quick_train_file') && exist(params.quick_train_file, 'file')
        try
            loaded_data = load(params.quick_train_file);
            if isfield(loaded_data, 'quick_params')
                fprintf('Found quick training parameters, loading...\n');
                params.quick_params = loaded_data.quick_params;
                fprintf('Quick training parameters loaded successfully!\n');
            end
        catch ME
            fprintf('Failed to load quick training parameters: %s\n', ME.message);
        end
    end
end
%%
function new_params = merge_parameters(new_params, old_params)
    % 合并参数，确保新版本的字段不被覆盖
    
    % 获取两个参数结构体的所有字段
    new_fields = fieldnames(new_params);
    old_fields = fieldnames(old_params);
    
    % 首先，将旧参数的所有字段复制到新参数
    for i = 1:length(old_fields)
        field_name = old_fields{i};
        if isfield(new_params, field_name)
            % 如果新参数也有这个字段，使用旧参数的值（但排除某些特定字段）
            if ~strcmp(field_name, 'training_stats') && ...
               ~strcmp(field_name, 'quick_train_file') && ...
               ~strcmp(field_name, 'quick_params')
                new_params.(field_name) = old_params.(field_name);
            end
        else
            % 如果新参数没有这个字段，添加它
            new_params.(field_name) = old_params.(field_name);
        end
    end
    
    % 确保新版本特有的字段存在
    if ~isfield(new_params, 'quick_train_file')
        new_params.quick_train_file = 'shanghai_quick_model.mat';
    end
    
    if ~isfield(new_params, 'training_stats')
        new_params.training_stats = struct();
        new_params.training_stats.last_trained = 'Upgraded from old version';
        new_params.training_stats.samples_used = 0;
        new_params.training_stats.accuracy = 0;
    end
end
%%
function params = quick_train_model(params)
    % 快速训练模型 - 使用10组沪语和10组非沪语样本
    fprintf('\n=== Quick Model Training (10+10 samples) ===\n');
    
    % 选择训练数据文件夹
    train_dir = uigetdir('', 'Select training data folder (should contain shanghai and non_shanghai subfolders)');
    if isequal(train_dir, 0)
        fprintf('No training folder selected\n');
        return;
    end
    
    % 检查文件夹结构
    shanghai_dir = fullfile(train_dir, 'shanghai');
    non_shanghai_dir = fullfile(train_dir, 'non_shanghai');
    
    fprintf('Checking directories:\n');
    fprintf('  Main directory: %s\n', train_dir);
    fprintf('  Shanghai directory: %s - exists: %d\n', shanghai_dir, exist(shanghai_dir, 'dir'));
    fprintf('  Non-Shanghai directory: %s - exists: %d\n', non_shanghai_dir, exist(non_shanghai_dir, 'dir'));
    
    if ~exist(shanghai_dir, 'dir') || ~exist(non_shanghai_dir, 'dir')
        fprintf('Error: Training folder should contain shanghai and non_shanghai subfolders\n');
        return;
    end
    
    % 获取训练文件
    shanghai_files = dir(fullfile(shanghai_dir, '*.wav'));
    non_shanghai_files = dir(fullfile(non_shanghai_dir, '*.wav'));
    
    fprintf('Files found:\n');
    fprintf('  Shanghai files: %d\n', length(shanghai_files));
    fprintf('  Non-Shanghai files: %d\n', length(non_shanghai_files));
    
    if isempty(shanghai_files) || isempty(non_shanghai_files)
        fprintf('Error: Not enough training files found\n');
        return;
    end
    
    % 限制每个类别最多10个样本
    num_shanghai = min(10, length(shanghai_files));
    num_non_shanghai = min(10, length(non_shanghai_files));
    
    fprintf('Using %d Shanghai samples and %d non-Shanghai samples for quick training\n', num_shanghai, num_non_shanghai);
    
    % 提取特征和标签
    features = [];
    labels = [];
    shanghai_features = [];
    non_shanghai_features = [];
    
    % 处理沪语样本
    fprintf('Processing Shanghai samples...\n');
    for i = 1:num_shanghai
        filename = fullfile(shanghai_dir, shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                shanghai_features = [shanghai_features; audio_features];
                labels = [labels; 1];
                fprintf('Shanghai sample %d/%d: %s\n', i, num_shanghai, shanghai_files(i).name);
            end
        catch ME
            fprintf('Error processing file %s: %s\n', shanghai_files(i).name, ME.message);
        end
    end
    
    % 处理非沪语样本
    fprintf('Processing non-Shanghai samples...\n');
    for i = 1:num_non_shanghai
        filename = fullfile(non_shanghai_dir, non_shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                non_shanghai_features = [non_shanghai_features; audio_features];
                labels = [labels; 0];
                fprintf('Non-Shanghai sample %d/%d: %s\n', i, num_non_shanghai, non_shanghai_files(i).name);
            end
        catch ME
            fprintf('Error processing file %s: %s\n', non_shanghai_files(i).name, ME.message);
        end
    end
    
    if isempty(features) || size(shanghai_features, 1) == 0 || size(non_shanghai_features, 1) == 0
        fprintf('Error: Failed to extract enough valid features\n');
        return;
    end
    
    fprintf('Feature extraction completed, total %d samples\n', size(features, 1));
    
    % 计算特征统计信息
    quick_params = calculate_quick_parameters(shanghai_features, non_shanghai_features);
    
    % 保存快速训练参数
    params.quick_params = quick_params;
    save(params.quick_train_file, 'quick_params');
    
    % 更新训练统计
    params.training_stats.last_trained = datestr(now);
    params.training_stats.samples_used = size(features, 1);
    params.training_stats.quick_training = true;
    
    fprintf('Quick training completed! Parameters saved to: %s\n', params.quick_train_file);
    fprintf('Use option 4 to apply these parameters to the detection system\n');
    
    % 显示训练结果摘要
    display_quick_training_summary(quick_params, shanghai_features, non_shanghai_features);
end
%%
function quick_params = calculate_quick_parameters(shanghai_features, non_shanghai_features)
    % 基于10+10样本计算优化的参数
    
    quick_params = struct();
    
    % 计算每个特征的均值和标准差
    for i = 1:4  % 现在只有4个特征
        feature_name = get_feature_name(i);
        
        % 沪语样本统计
        shanghai_vals = shanghai_features(:, i);
        non_shanghai_vals = non_shanghai_features(:, i);
        
        quick_params.(['shanghai_mean_', feature_name]) = mean(shanghai_vals);
        quick_params.(['shanghai_std_', feature_name]) = std(shanghai_vals);
        quick_params.(['non_shanghai_mean_', feature_name]) = mean(non_shanghai_vals);
        quick_params.(['non_shanghai_std_', feature_name]) = std(non_shanghai_vals);
        
        % 计算分离度
        mean_diff = abs(quick_params.(['shanghai_mean_', feature_name]) - ...
                       quick_params.(['non_shanghai_mean_', feature_name]));
        pooled_std = (quick_params.(['shanghai_std_', feature_name]) + ...
                     quick_params.(['non_shanghai_std_', feature_name])) / 2;
        
        if pooled_std > 0
            quick_params.(['separation_', feature_name]) = mean_diff / pooled_std;
        else
            quick_params.(['separation_', feature_name]) = 0;
        end
    end
    
    % 基于分离度重新计算权重
    separations = [quick_params.separation_low_freq, ...
                  quick_params.separation_pitch_var, ...
                  quick_params.separation_zcr, ...
                  quick_params.separation_energy_var];
    
    total_separation = sum(separations);
    if total_separation > 0
        quick_params.optimized_weights = separations / total_separation;
    else
        quick_params.optimized_weights = [0.4, 0.3, 0.2, 0.1]; % 默认权重
    end
    
    % 计算优化的阈值
    quick_params.optimized_threshold = 0.6; % 可以根据数据调整
    
    fprintf('Feature weights calculated based on training data:\n');
    feature_names = {'Low Frequency', 'Pitch Variation', 'Zero Crossing', 'Energy Variation'};
    for i = 1:4
        fprintf('  %s: %.3f (separation: %.3f)\n', feature_names{i}, ...
                quick_params.optimized_weights(i), separations(i));
    end
end
%%
function feature_name = get_feature_name(index)
    % 获取特征名称
    switch index
        case 1
            feature_name = 'low_freq';
        case 2
            feature_name = 'pitch_var';
        case 3
            feature_name = 'zcr';
        case 4
            feature_name = 'energy_var';
        otherwise
            feature_name = 'unknown';
    end
end
%%
function display_quick_training_summary(quick_params, shanghai_features, non_shanghai_features)
    % 显示快速训练摘要
    fprintf('\n=== Quick Training Results Summary ===\n');
    fprintf('Shanghai samples: %d\n', size(shanghai_features, 1));
    fprintf('Non-Shanghai samples: %d\n', size(non_shanghai_features, 1));
    
    feature_names = {'Low Frequency', 'Pitch Variation', 'Zero Crossing', 'Energy Variation'};
    
    fprintf('\nFeature Statistics Comparison:\n');
    fprintf('%-15s %-10s %-10s %-10s\n', 'Feature', 'Shanghai', 'Non-Shanghai', 'Separation');
    fprintf('%-15s %-10s %-10s %-10s\n', '-------', '--------', '------------', '----------');
    
    for i = 1:4
        feature_name = get_feature_name(i);
        shanghai_mean = quick_params.(['shanghai_mean_', feature_name]);
        non_shanghai_mean = quick_params.(['non_shanghai_mean_', feature_name]);
        separation = quick_params.(['separation_', feature_name]);
        
        fprintf('%-15s %-10.3f %-10.3f %-10.3f\n', ...
                feature_names{i}, shanghai_mean, non_shanghai_mean, separation);
    end
    
    fprintf('\nOptimized Weight Parameters:\n');
    total_weight = 0;
    for i = 1:4
        weight = quick_params.optimized_weights(i);
        total_weight = total_weight + weight;
        fprintf('  %s: %.3f\n', feature_names{i}, weight);
    end
    fprintf('Total weight: %.3f\n', total_weight);
end
%%
function params = update_detection_parameters(params)
    % 更新检测参数
    fprintf('\n=== Update Detection Parameters ===\n');
    
    if ~isfield(params, 'quick_params')
        fprintf('Error: No training parameters found, please perform quick training first\n');
        return;
    end
    
    quick_params = params.quick_params;
    
    % 显示当前参数和推荐参数
    fprintf('Current parameters vs Recommended parameters:\n');
    fprintf('%-15s %-8s %-8s\n', 'Parameter', 'Current', 'Recommended');
    fprintf('%-15s %-8s %-8s\n', '---------', '-------', '------------');
    
    current_weights = [params.low_freq_weight, params.pitch_var_weight, ...
                      params.zcr_weight, params.energy_var_weight];
    
    feature_names = {'Low Freq Weight', 'Pitch Weight', 'ZCR Weight', 'Energy Weight'};
    
    for i = 1:4
        fprintf('%-15s %-8.3f %-8.3f\n', feature_names{i}, ...
                current_weights(i), quick_params.optimized_weights(i));
    end
    
    fprintf('%-15s %-8.3f %-8.3f\n', 'Threshold', params.shanghai_threshold, quick_params.optimized_threshold);
    
    % 询问用户是否更新
    choice = input('Apply recommended parameters? (y/n): ', 's');
    
    if strcmpi(choice, 'y') || strcmpi(choice, 'yes')
        % 更新权重参数
        params.low_freq_weight = quick_params.optimized_weights(1);
        params.pitch_var_weight = quick_params.optimized_weights(2);
        params.zcr_weight = quick_params.optimized_weights(3);
        params.energy_var_weight = quick_params.optimized_weights(4);
        
        % 更新阈值
        params.shanghai_threshold = quick_params.optimized_threshold;
        
        % 保存更新后的参数
        save(params.model_file, 'params');
        
        fprintf('Parameters updated successfully!\n');
        fprintf('New weights: [%.3f, %.3f, %.3f, %.3f]\n', ...
                params.low_freq_weight, params.pitch_var_weight, ...
                params.zcr_weight, params.energy_var_weight);
        fprintf('New threshold: %.3f\n', params.shanghai_threshold);
        
        % 更新训练统计
        params.training_stats.parameters_updated = datestr(now);
        params.training_stats.using_optimized_params = true;
    else
        fprintf('Parameters not updated, continuing with current parameters\n');
    end
end
%%
function display_current_parameters(params)
    % 显示当前参数
    fprintf('\n=== Current Detection Parameters ===\n');
    
    fprintf('Basic Parameters:\n');
    fprintf('  Sampling rate: %d Hz\n', params.fs);
    fprintf('  Frame length: %.3f s\n', params.frame_len);
    fprintf('  Hop length: %.3f s\n', params.hop_len);
    fprintf('  Minimum audio duration: %.1f s\n', params.min_duration);
    
    fprintf('\nDetection Thresholds and Weights:\n');
    fprintf('  Shanghai dialect threshold: %.3f\n', params.shanghai_threshold);
    fprintf('  Low frequency weight: %.3f\n', params.low_freq_weight);
    fprintf('  Pitch variation weight: %.3f\n', params.pitch_var_weight);
    fprintf('  Zero crossing weight: %.3f\n', params.zcr_weight);
    fprintf('  Energy variation weight: %.3f\n', params.energy_var_weight);
    
    if isfield(params, 'training_stats')
        fprintf('\nTraining Statistics:\n');
        fprintf('  Last trained: %s\n', params.training_stats.last_trained);
        fprintf('  Samples used: %d\n', params.training_stats.samples_used);
        if isfield(params.training_stats, 'using_optimized_params') && params.training_stats.using_optimized_params
            fprintf('  Currently using optimized parameters: Yes\n');
        else
            fprintf('  Currently using optimized parameters: No\n');
        end
    end
    
    if isfield(params, 'quick_params')
        fprintf('\nQuick training parameters available: Yes\n');
        fprintf('  Use option 4 to apply these parameters\n');
    else
        fprintf('\nQuick training parameters available: No\n');
        fprintf('  Use option 3 to perform quick training\n');
    end
end
%%
function train_model(params)
    % 训练沪语检测模型
    fprintf('\n=== Shanghai Dialect Detection Model Training ===\n');
    
    % 选择训练数据文件夹
    train_dir = uigetdir('', 'Select training data folder (should contain shanghai and non_shanghai subfolders)');
    if isequal(train_dir, 0)
        fprintf('No training folder selected\n');
        return;
    end
    
    % 检查文件夹结构
    shanghai_dir = fullfile(train_dir, 'shanghai');
    non_shanghai_dir = fullfile(train_dir, 'non_shanghai');
    
    if ~exist(shanghai_dir, 'dir') || ~exist(non_shanghai_dir, 'dir')
        fprintf('Error: Training folder should contain shanghai and non_shanghai subfolders\n');
        return;
    end
    
    % 获取训练文件
    shanghai_files = dir(fullfile(shanghai_dir, '*.wav'));
    non_shanghai_files = dir(fullfile(non_shanghai_dir, '*.wav'));
    
    if isempty(shanghai_files) && isempty(non_shanghai_files)
        fprintf('Error: No WAV format training files found\n');
        return;
    end
    
    fprintf('Found %d Shanghai samples and %d non-Shanghai samples\n', ...
            length(shanghai_files), length(non_shanghai_files));
    
    % 提取特征和标签
    features = [];
    labels = [];
    
    % 处理沪语样本
    fprintf('Processing Shanghai samples...\n');
    for i = 1:length(shanghai_files)
        filename = fullfile(shanghai_dir, shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                labels = [labels; 1];  % 沪语标签为1
                fprintf('Processing Shanghai sample %d/%d: %s\n', i, length(shanghai_files), shanghai_files(i).name);
            end
        catch ME
            fprintf('Error processing file %s: %s\n', shanghai_files(i).name, ME.message);
        end
    end
    
    % 处理非沪语样本
    fprintf('Processing non-Shanghai samples...\n');
    for i = 1:length(non_shanghai_files)
        filename = fullfile(non_shanghai_dir, non_shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                labels = [labels; 0];  % 非沪语标签为0
                fprintf('Processing non-Shanghai sample %d/%d: %s\n', i, length(non_shanghai_files), non_shanghai_files(i).name);
            end
        catch ME
            fprintf('Error processing file %s: %s\n', non_shanghai_files(i).name, ME.message);
        end
    end
    
    if isempty(features)
        fprintf('Error: Failed to extract any valid features\n');
        return;
    end
    
    fprintf('Feature extraction completed, total %d samples\n', size(features, 1));
    
    % 训练分类器
    fprintf('Training classifier...\n');
    model = train_classifier(features, labels);
    
    % 保存模型
    save(params.model_file, 'model', 'params');
    fprintf('Model saved to: %s\n', params.model_file);
    
    % 显示训练结果
    display_training_results(model, features, labels);
end
%%
function features = extract_features_from_file(filename, params)
    try
        % 读取音频文件
        [audio, fs] = audioread(filename);
        if size(audio, 2) > 1
            audio = mean(audio, 2); % 转为单声道
        end
        
        % 检查音频长度
        if length(audio)/fs < params.min_duration
            fprintf('Audio too short, skipping: %s\n', filename);
            features = [];
            return;
        end
        
        % 重采样到16kHz（如果需要）
        if fs ~= params.fs
            audio = resample(audio, params.fs, fs);
            fs = params.fs;
        end
        
        % 提取关键特征
        audio_features = extract_shanghai_features(audio, fs, params);
        
        % 转换为特征向量 - 添加健壮性检查
        if isstruct(audio_features)
            % 确保所有必需的字段都存在
            required_fields = {'low_freq_ratio', 'pitch_variance', 'zcr', 'energy_variance'};
            
            % 检查缺失的字段并提供默认值
            for i = 1:length(required_fields)
                if ~isfield(audio_features, required_fields{i})
                    fprintf('Warning: Feature field %s missing, using default value\n', required_fields{i});
                    switch required_fields{i}
                        case 'low_freq_ratio'
                            audio_features.low_freq_ratio = 0.3;
                        case 'pitch_variance'
                            audio_features.pitch_variance = 35;
                        case 'zcr'
                            audio_features.zcr = 0.08;
                        case 'energy_variance'
                            audio_features.energy_variance = 0.1;
                    end
                end
            end
            
            % 创建特征向量
            features = [audio_features.low_freq_ratio, ...
                       audio_features.pitch_variance, ...
                       audio_features.zcr, ...
                       audio_features.energy_variance];
            
            % 检查特征是否有效（没有NaN或Inf）
            if any(isnan(features)) || any(isinf(features))
                fprintf('Warning: File %s contains invalid features, skipping\n', filename);
                features = [];
            end
        else
            fprintf('Warning: Feature extraction failed for file %s\n', filename);
            features = [];
        end
        
    catch ME
        fprintf('Error processing file %s: %s\n', filename, ME.message);
        features = [];
    end
end
%%
function model = train_classifier(features, labels)
    % 训练分类器模型
    
    % 数据标准化
    model.feature_mean = mean(features);
    model.feature_std = std(features);
    features_normalized = (features - model.feature_mean) ./ model.feature_std;
    
    % 替换NaN值为0
    features_normalized(isnan(features_normalized)) = 0;
    
    % 使用逻辑回归训练简单分类器
    try
        model.classifier = fitclinear(features_normalized, labels, ...
                                     'Learner', 'logistic', ...
                                     'Regularization', 'ridge');
        model.model_type = 'logistic_regression';
    catch
        % 如果fitclinear不可用，使用简单的阈值分类器
        fprintf('Linear classifier failed, using threshold classifier\n');
        model = train_threshold_classifier(features, labels);
        model.model_type = 'threshold';
    end
    
    % 保存特征重要性（基于相关系数）- 修复NaN问题
    feature_importance = zeros(1, size(features, 2));
    for i = 1:size(features, 2)
        try
            [r, ~] = corr(features(:, i), labels);
            if ~isnan(r)
                feature_importance(i) = abs(r);
            else
                feature_importance(i) = 0;
            end
        catch
            feature_importance(i) = 0;
        end
    end
    
    model.feature_importance = feature_importance;
end
%%
function model = train_threshold_classifier(features, labels)
    % 训练简单的阈值分类器
    
    % 计算每个特征的最佳阈值
    num_features = size(features, 2);
    model.thresholds = zeros(1, num_features);
    model.directions = zeros(1, num_features); % 1表示大于阈值为正类，-1表示小于阈值为正类
    model.weights = zeros(1, num_features);
    
    for i = 1:num_features
        feature_vals = features(:, i);
        pos_vals = feature_vals(labels == 1);
        neg_vals = feature_vals(labels == 0);
        
        if isempty(pos_vals) || isempty(neg_vals)
            model.thresholds(i) = median(feature_vals);
            model.directions(i) = 1;
            model.weights(i) = 0.25;
            continue;
        end
        
        % 计算两类分布的均值
        pos_mean = mean(pos_vals);
        neg_mean = mean(neg_vals);
        
        % 设置阈值和方向
        model.thresholds(i) = (pos_mean + neg_mean) / 2;
        if pos_mean > neg_mean
            model.directions(i) = 1;  % 沪语样本该特征值较大
        else
            model.directions(i) = -1; % 沪语样本该特征值较小
        end
        
        % 基于分离度设置权重
        separation = abs(pos_mean - neg_mean) / (std(pos_vals) + std(neg_vals));
        model.weights(i) = separation;
    end
    
    % 归一化权重
    if sum(model.weights) > 0
        model.weights = model.weights / sum(model.weights);
    else
        model.weights = ones(1, num_features) / num_features;
    end
    
    model.feature_mean = mean(features);
    model.feature_std = std(features);
end
%%
function predictions = predict_with_threshold(model, features)
    % 使用阈值分类器进行预测
    scores = zeros(size(features, 1), 1);
    
    for i = 1:size(features, 1)
        score = 0;
        for j = 1:size(features, 2)
            if model.directions(j) == 1
                % 特征值大于阈值时加分
                if features(i, j) > model.thresholds(j)
                    score = score + model.weights(j);
                end
            else
                % 特征值小于阈值时加分
                if features(i, j) < model.thresholds(j)
                    score = score + model.weights(j);
                end
            end
        end
        scores(i) = score;
    end
    
    % 使用0.5作为分类阈值
    predictions = scores >= 0.5;
end
%%
function detect_audio_file(params)
    % 检测音频文件（原有代码保持不变）
    [file, path] = uigetfile({'*.wav;*.mp3', 'Audio Files'}, 'Select audio file to detect');
    if isequal(file, 0)
        fprintf('No file selected\n');
        return;
    end
    
    filename = fullfile(path, file);
    fprintf('Detecting: %s\n', file);
    
    try
        % 读取音频
        [audio, fs] = audioread(filename);
        if size(audio, 2) > 1
            audio = mean(audio, 2); % 转为单声道
        end
        
        duration = length(audio) / fs;
        fprintf('Audio info: %.2f seconds, %d Hz\n', duration, fs);
        
        % 检查音频长度
        if duration < params.min_duration
            fprintf('Audio too short, minimum %.1f seconds required\n', params.min_duration);
            return;
        end
        
        % 重采样到16kHz（如果需要）
        if fs ~= params.fs
            audio = resample(audio, params.fs, fs);
            fs = params.fs;
        end
        
        % 提取关键特征
        features = extract_shanghai_features(audio, fs, params);
        
        % 沪语检测
        [is_shanghai, confidence, score] = detect_shanghai(features, params);
        
        % 显示结果
        display_detection_result(is_shanghai, confidence, score, features, audio, fs);
        
    catch ME
        fprintf('Error during detection: %s\n', ME.message);
    end
end
%%
function features = extract_shanghai_features(audio, fs, params)
    % 提取沪语关键特征
    fprintf('Extracting Shanghai dialect features...\n');
    
    features = struct();
    
    % 1. 低频能量比（沪语低频成分较多）
    features.low_freq_ratio = get_low_frequency_ratio(audio, fs);
    
    % 2. 基频特征（沪语音调变化特征）
    [pitch_mean, pitch_variance] = get_pitch_features(audio, fs, params);
    features.pitch_mean = pitch_mean;
    features.pitch_variance = pitch_variance;
    
    % 3. 过零率（沪语辅音特征）
    features.zcr = get_zero_crossing_rate(audio);
    
    % 4. 能量变化
    features.energy_variance = get_energy_variance(audio, fs, params);
    
    fprintf('Feature extraction completed\n');
end
%%
function low_freq_ratio = get_low_frequency_ratio(audio, fs)
    % 计算低频能量比（80-400Hz）
    N = min(1024, length(audio));
    if N < 256
        low_freq_ratio = 0.3;
        return;
    end
    
    frame = audio(1:N);
    windowed = frame .* hamming(N);
    Y = fft(windowed, N);
    magnitude = abs(Y(1:N/2));
    f = (0:N/2-1) * fs / N;
    
    % 低频范围：80-400Hz（沪语特征频率）
    low_freq_mask = (f >= 80 & f <= 400);
    total_energy = sum(magnitude);
    
    if total_energy > 0
        low_freq_ratio = sum(magnitude(low_freq_mask)) / total_energy;
    else
        low_freq_ratio = 0.3;
    end
end
%%
function [pitch_mean, pitch_variance] = get_pitch_features(audio, fs, params)
    % 提取基频特征
    frame_size = round(params.frame_len * fs);
    hop_size = round(params.hop_len * fs);
    
    num_frames = floor((length(audio) - frame_size) / hop_size) + 1;
    pitches = [];
    
    % 分析前30帧以避免计算过长
    max_frames = min(30, num_frames);
    
    for i = 1:max_frames
        start_idx = (i-1) * hop_size + 1;
        end_idx = start_idx + frame_size - 1;
        
        if end_idx > length(audio)
            break;
        end
        
        frame = audio(start_idx:end_idx);
        pitch = estimate_pitch_simple(frame, fs);
        
        % 有效基频范围：80-400Hz
        if pitch > 80 && pitch < 400
            pitches = [pitches, pitch];
        end
    end
    
    if length(pitches) >= 3
        pitch_mean = mean(pitches);
        pitch_variance = std(pitches);
    else
        % 默认值（基于沪语特征）
        pitch_mean = 180;
        pitch_variance = 35;
    end
end
%%
function pitch = estimate_pitch_simple(frame, fs)
    % 简化的基频估计
    frame = frame - mean(frame);
    
    % 自相关法
    r = xcorr(frame, 'coeff');
    r = r(length(frame):end);
    
    % 寻找峰值
    [peaks, locs] = findpeaks(r);
    if length(peaks) >= 2
        % 避免零延迟峰值，找第一个显著峰值
        threshold = max(peaks) * 0.3;
        valid_peaks = peaks > threshold;
        valid_locs = locs(valid_peaks);
        
        if length(valid_locs) >= 2
            fundamental_period = valid_locs(2) - valid_locs(1);
            if fundamental_period > 0
                pitch = fs / fundamental_period;
                return;
            end
        end
    end
    
    pitch = 0;
end
%%
function zcr = get_zero_crossing_rate(audio)
    % 计算过零率
    zcr = sum(abs(diff(audio > 0))) / length(audio);
end
%%
function energy_variance = get_energy_variance(audio, fs, params)
    % 计算能量变化
    frame_size = round(params.frame_len * fs);
    hop_size = round(params.hop_len * fs);
    
    num_frames = floor((length(audio) - frame_size) / hop_size) + 1;
    energies = zeros(1, min(20, num_frames));
    
    for i = 1:length(energies)
        start_idx = (i-1) * hop_size + 1;
        end_idx = start_idx + frame_size - 1;
        
        if end_idx > length(audio)
            break;
        end
        
        frame = audio(start_idx:end_idx);
        energies(i) = mean(frame.^2);
    end
    
    % 归一化
    if max(energies) > 0
        energies = energies / max(energies);
    end
    
    energy_variance = var(energies);
end
%%
function [is_shanghai, confidence, total_score] = detect_shanghai(features, params)
    % 沪语检测逻辑
    
    % 各个特征的得分（0-1）
    scores = zeros(1, 4);
    
    % 放宽版：
    scores(1) = min(features.low_freq_ratio * 4, 1.0);      % 乘数从3增加到4
    scores(2) = min(features.pitch_variance / 50, 1.0);     % 除数从60降到50
    scores(3) = max(0, 1 - features.zcr / 0.15);            % 除数从0.12升到0.15
    scores(4) = min(features.energy_variance * 10, 1.0);    % 乘数从8增加到10
    
    % 加权总分
    weights = [params.low_freq_weight, params.pitch_var_weight, ...
               params.zcr_weight, params.energy_var_weight];
    
    total_score = sum(scores .* weights);
    
    % 判断结果
    is_shanghai = total_score >= params.shanghai_threshold;
    
    % 置信度计算
    if is_shanghai
        confidence = min((total_score - params.shanghai_threshold) / ...
                        (1 - params.shanghai_threshold), 0.95);
    else
        confidence = min((params.shanghai_threshold - total_score) / ...
                        params.shanghai_threshold, 0.95);
    end
    
    fprintf('Detection score: %.3f, Threshold: %.3f\n', total_score, params.shanghai_threshold);
end
%%
function display_detection_result(is_shanghai, confidence, score, features, audio, fs)
    % 显示检测结果
    fprintf('\n=== Detection Results ===\n');
    
    if is_shanghai
        fprintf('Detection result: Shanghai Dialect\n');
        fprintf('Confidence: %.1f%%\n', confidence * 100);
    else
        fprintf('Detection result: Other Language\n');
        fprintf('Confidence: %.1f%%\n', confidence * 100);
    end
    
    fprintf('Overall score: %.3f/1.000\n', score);
    
    fprintf('\n=== Feature Details ===\n');
    fprintf('Low frequency ratio: %.3f (Shanghai>0.3)\n', features.low_freq_ratio);
    fprintf('Average pitch: %.1f Hz\n', features.pitch_mean);
    fprintf('Pitch variation: %.1f Hz (Shanghai>25)\n', features.pitch_variance);
    fprintf('Zero crossing rate: %.3f (Shanghai<0.1)\n', features.zcr);
    fprintf('Energy variation: %.3f\n', features.energy_variance);
    
    % 绘制结果图表
    plot_detection_results(is_shanghai, score, features, audio, fs);
end
%%
function plot_detection_results(is_shanghai, score, features, audio, fs)
    % 绘制检测结果图表 - 使用更简单的布局确保所有内容都显示
    figure('Name', 'Shanghai Dialect Detection Analysis', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 900]);
    
    % 使用2x2布局，给结果留出更大空间
    % 1. 音频波形
    subplot(2, 2, 1);
    t = (0:length(audio)-1) / fs;
    plot(t, audio, 'b', 'LineWidth', 1);
    title('Audio Waveform', 'FontSize', 12);
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;
    
    % 2. 频谱分析
    subplot(2, 2, 2);
    N = min(1024, length(audio));
    Y = fft(audio(1:N), N);
    f = (0:N-1) * fs / N;
    plot(f(1:N/2), abs(Y(1:N/2)), 'r', 'LineWidth', 1);
    title('Spectral Analysis', 'FontSize', 12);
    xlabel('Frequency (Hz)'); ylabel('Magnitude'); grid on;
    
    % 标记沪语特征频率区域
    hold on;
    yl = ylim;
    fill([80, 400, 400, 80], [yl(1), yl(1), yl(2), yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    legend('Spectrum', 'Shanghai Feature Band', 'Location', 'northeast');
    
    % 3. 特征条形图
    subplot(2, 2, 3);
    feature_names = {'Low Frequency', 'Pitch Variation', 'Zero Crossing', 'Energy Variation'};
    feature_values = [features.low_freq_ratio, features.pitch_variance/60, ...
                     1 - min(features.zcr/0.15, 1), ...
                     min(features.energy_variance*10, 1)];
    
    bar(feature_values, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', feature_names, 'FontSize', 10);
    title('Normalized Feature Values', 'FontSize', 12); 
    ylabel('Value'); 
    grid on;
    ylim([0 1]);
    
    % 4. 主要结果显示 - 使用整个右侧空间
    subplot(2, 2, 4);
    axis off;
    
    % 背景颜色区分
    if is_shanghai
        bg_color = [1, 0.9, 0.9];  % 浅红色背景
        result_color = 'red';
    else
        bg_color = [0.9, 0.95, 1];  % 浅蓝色背景
        result_color = 'blue';
    end
    
    % 设置背景
    set(gca, 'Color', bg_color);
    
    % 主标题
    text(0.5, 0.95, 'Shanghai Dialect Detection Final Result', 'FontSize', 20, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', [0, 0.4, 0.6]);
    
    % 分隔线
    line([0.1, 0.9], [0.90, 0.90], 'Color', 'k', 'LineWidth', 2);
    
    % 显著显示检测结果
    if is_shanghai
        % 上海话结果 - 红色突出显示
        text(0.5, 0.82, 'SHANGHAI DIALECT', 'FontSize', 28, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'red');
        text(0.5, 0.75, 'Shanghai dialect detected', 'FontSize', 18, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'red');
    else
        % 其他语言结果 - 蓝色显示
        text(0.5, 0.82, 'OTHER LANGUAGE', 'FontSize', 28, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'blue');
        text(0.5, 0.75, 'No Shanghai dialect features detected', 'FontSize', 18, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'blue');
    end
    
    % 置信度显示 - 位置上移
    text(0.5, 0.65, sprintf('Confidence: %.1f%%', score * 100), 'FontSize', 20, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', result_color);
    
    % 得分信息框
    rectangle('Position', [0.2, 0.52, 0.6, 0.08], 'FaceColor', [1, 1, 1], ...
              'EdgeColor', 'k', 'LineWidth', 2);
    text(0.5, 0.57, sprintf('Overall Score: %.3f', score), 'FontSize', 16, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    text(0.5, 0.54, sprintf('Threshold: 0.600'), 'FontSize', 14, ...
         'HorizontalAlignment', 'center');
    
    % 判断状态
    if is_shanghai
        status_text = '✓ Exceeds threshold, identified as Shanghai dialect';
        status_color = 'green';
    else
        status_text = '✗ Below threshold, identified as other language';
        status_color = 'red';
    end
    text(0.5, 0.42, status_text, 'FontSize', 14, 'HorizontalAlignment', 'center', ...
         'Color', status_color, 'FontWeight', 'bold');
    
    % 特征匹配情况
    text(0.5, 0.35, 'Feature Matching Status:', 'FontSize', 12, 'HorizontalAlignment', 'center', ...
         'FontWeight', 'bold');
    
    % 显示各个特征的匹配状态
    feature_status = {
        sprintf('Low frequency: %.3f %s', features.low_freq_ratio, iif(features.low_freq_ratio > 0.3, '✓', '✗')),
        sprintf('Pitch variation: %.1f Hz %s', features.pitch_variance, iif(features.pitch_variance > 25, '✓', '✗')),
        sprintf('Zero crossing: %.3f %s', features.zcr, iif(features.zcr < 0.12, '✓', '✗'))
    };
    
    for i = 1:length(feature_status)
        text(0.5, 0.30 - (i-1)*0.05, feature_status{i}, 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
    end
    
    % 音频基本信息
    text(0.5, 0.05, sprintf('Audio duration: %.1f seconds | Sampling rate: %d Hz', length(audio)/fs, fs), ...
         'FontSize', 10, 'HorizontalAlignment', 'center', 'Color', [0.5, 0.5, 0.5]);
    
    % 在整个图的上方添加总标题
    sgtitle(sprintf('Shanghai Dialect Detection Analysis - Final Result: %s (Score: %.3f)', ...
                   iif(is_shanghai, 'Shanghai Dialect', 'Other Language'), score), ...
            'FontSize', 16, 'FontWeight', 'bold', ...
            'Color', iif(is_shanghai, 'red', 'blue'));
end
%%
function result = iif(condition, true_val, false_val)
    % 简化的条件判断函数
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
%%
function display_training_results(model, features, labels)
    % 显示训练结果 - 修复版
    fprintf('\n=== Training Results ===\n');
    
    % 预测训练集
    if strcmp(model.model_type, 'logistic_regression')
        features_normalized = (features - model.feature_mean) ./ model.feature_std;
        features_normalized(isnan(features_normalized)) = 0;
        predictions = predict(model.classifier, features_normalized);
        
        % 对于线性分类器，计算决策分数而不是概率
        decision_scores = predict(model.classifier, features_normalized, 'ObservationsIn', 'rows');
        % 将决策分数转换为0-1范围的置信度分数
        confidence_scores = 1 ./ (1 + exp(-decision_scores));
        
    else
        predictions = predict_with_threshold(model, features);
        % 对于阈值分类器，直接使用预测得分
        scores = zeros(size(features, 1), 1);
        for i = 1:size(features, 1)
            score = 0;
            for j = 1:size(features, 2)
                if model.directions(j) == 1
                    if features(i, j) > model.thresholds(j)
                        score = score + model.weights(j);
                    end
                else
                    if features(i, j) < model.thresholds(j)
                        score = score + model.weights(j);
                    end
                end
            end
            scores(i) = score;
        end
        confidence_scores = scores;
    end
    
    % 计算准确率
    accuracy = sum(predictions == labels) / length(labels);
    fprintf('Training set accuracy: %.2f%%\n', accuracy * 100);
    
    % 计算混淆矩阵
    TP = sum(predictions == 1 & labels == 1);
    FP = sum(predictions == 1 & labels == 0);
    TN = sum(predictions == 0 & labels == 0);
    FN = sum(predictions == 0 & labels == 1);
    
    fprintf('Confusion Matrix:\n');
    fprintf('          Predicted Shanghai  Predicted Other\n');
    fprintf('Actual Shanghai    %3d            %3d\n', TP, FN);
    fprintf('Actual Other       %3d            %3d\n', FP, TN);
    
    % 计算其他指标
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1_score = 2 * (precision * recall) / (precision + recall);
    else
        f1_score = 0;
    end
    
    fprintf('\nDetailed Metrics:\n');
    fprintf('Precision: %.3f\n', precision);
    fprintf('Recall: %.3f\n', recall);
    fprintf('F1 Score: %.3f\n', f1_score);
    
    % 显示特征重要性
    fprintf('\nFeature Importance:\n');
    feature_names = {'Low Frequency', 'Pitch Variation', 'Zero Crossing', 'Energy Variation'};
    for i = 1:length(feature_names)
        fprintf('%s: %.3f\n', feature_names{i}, model.feature_importance(i));
    end
    
    % 创建综合可视化图表
    create_training_visualization(model, features, labels, predictions, ...
                                 confidence_scores, accuracy, precision, ...
                                 recall, f1_score, TP, FP, TN, FN);
end
%%
function create_training_visualization(model, features, labels, ~, ...
                                     confidence_scores, accuracy, precision, ...
                                     recall, f1_score, TP, FP, TN, FN)
    % 创建训练结果综合可视化
    
    figure('Name', 'Shanghai Dialect Detection Model Training Results', 'NumberTitle', 'off', ...
           'Position', [100, 50, 1400, 900]);
    
    % 1. 性能指标雷达图
    subplot(2, 3, 1);
    performance_metrics = [accuracy, precision, recall, f1_score];
    metric_names = {'Accuracy', 'Precision', 'Recall', 'F1 Score'};
    
    % 创建雷达图
    angles = linspace(0, 2*pi, length(performance_metrics) + 1);
    angles = angles(1:end-1);
    
    polarplot([angles, angles(1)], [performance_metrics, performance_metrics(1)], ...
              'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 8);
    thetaticks(angles * 180/pi);
    thetaticklabels(metric_names);
    rlim([0 1]);
    title('Performance Metrics Radar Chart', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    % 添加数值标签
    for i = 1:length(performance_metrics)
        text(angles(i), performance_metrics(i) + 0.05, ...
             sprintf('%.3f', performance_metrics(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
    
    % 2. 混淆矩阵热图
    subplot(2, 3, 2);
    confusion_mat = [TP, FN; FP, TN];
    heatmap_data = confusion_mat ./ sum(confusion_mat(:)) * 100;
    
    imagesc(heatmap_data);
    colormap(flipud(gray));
    colorbar;
    
    % 设置坐标轴
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Predicted Shanghai', 'Predicted Other'});
    set(gca, 'YTick', 1:2, 'YTickLabel', {'Actual Shanghai', 'Actual Other'});
    set(gca, 'XTickLabelRotation', 45);
    
    % 添加数值文本
    for i = 1:2
        for j = 1:2
            text(j, i, sprintf('%d\n(%.1f%%)', confusion_mat(i,j), heatmap_data(i,j)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontWeight', 'bold', 'Color', 'white', 'FontSize', 11);
        end
    end
    
    title('Confusion Matrix Heatmap (%)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 3. 特征重要性条形图
    subplot(2, 3, 3);
    feature_names = {'Low Frequency', 'Pitch Variation', 'Zero Crossing', 'Energy Variation'};
    
    barh(model.feature_importance, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'black');
    set(gca, 'YTickLabel', feature_names);
    xlabel('Importance Score');
    title('Feature Importance', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    % 4. 分类结果散点图（前两个主要特征）- 修复索引超出范围问题
    subplot(2, 3, 4);
    if size(features, 2) >= 2
        % 找到最重要的两个特征，确保索引有效
        [~, important_idx] = sort(model.feature_importance, 'descend');
        
        % 确保有足够的有效特征
        valid_features = min(length(important_idx), 4);
        if valid_features >= 2
            feat1 = important_idx(1);
            feat2 = important_idx(2);
            
            % 确保索引在1-4范围内
            feat1 = min(max(feat1, 1), 4);
            feat2 = min(max(feat2, 1), 4);
            
            % 绘制散点图
            gscatter(features(:, feat1), features(:, feat2), labels, ...
                    [1 0 0; 0 0 1], 'o*', 15);
            xlabel(feature_names{feat1});
            ylabel(feature_names{feat2});
            legend('Non-Shanghai', 'Shanghai', 'Location', 'best');
            title('Feature Space Distribution (Top 2 Features)', 'FontSize', 12, 'FontWeight', 'bold');
            grid on;
        else
            text(0.5, 0.5, 'Insufficient features', 'HorizontalAlignment', 'center', ...
                 'FontSize', 12, 'FontWeight', 'bold');
            title('Feature Space Distribution', 'FontSize', 12, 'FontWeight', 'bold');
            axis off;
        end
    else
        text(0.5, 0.5, 'Insufficient feature dimensions', 'HorizontalAlignment', 'center', ...
             'FontSize', 12, 'FontWeight', 'bold');
        title('Feature Space Distribution', 'FontSize', 12, 'FontWeight', 'bold');
        axis off;
    end
    
    % 5. 分数分布直方图
    subplot(2, 3, 5);
    histogram(confidence_scores(labels==1), 'FaceColor', 'red', 'FaceAlpha', 0.6, 'BinWidth', 0.1);
    hold on;
    histogram(confidence_scores(labels==0), 'FaceColor', 'blue', 'FaceAlpha', 0.6, 'BinWidth', 0.1);
    xlabel('Confidence Score');
    ylabel('Frequency');
    title('Confidence Score Distribution', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Shanghai', 'Non-Shanghai');
    grid on;
    
    % 6. 训练摘要面板
    subplot(2, 3, 6);
    axis off;
    
    % 背景设置
    set(gca, 'Color', [0.95, 0.95, 0.95]);
    
    % 标题
    text(0.5, 0.95, 'Training Results Summary', 'FontSize', 16, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', [0, 0.4, 0.6]);
    
    % 分隔线
    line([0.1, 0.9], [0.90, 0.90], 'Color', 'k', 'LineWidth', 2);
    
    % 关键指标
    text(0.1, 0.80, sprintf('Model Type: %s', model.model_type), ...
         'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.75, sprintf('Total Samples: %d', length(labels)), ...
         'FontSize', 11);
    text(0.1, 0.70, sprintf('Shanghai Samples: %d', sum(labels==1)), ...
         'FontSize', 11);
    text(0.1, 0.65, sprintf('Non-Shanghai Samples: %d', sum(labels==0)), ...
         'FontSize', 11);
    
    % 性能指标框
    rectangle('Position', [0.1, 0.45, 0.8, 0.15], 'FaceColor', 'white', ...
              'EdgeColor', 'k', 'LineWidth', 2);
    
    text(0.5, 0.57, 'Performance Metrics', 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
    
    text(0.3, 0.52, sprintf('Accuracy: %.1f%%', accuracy*100), ...
         'FontSize', 11, 'FontWeight', 'bold', 'Color', 'blue');
    text(0.7, 0.52, sprintf('F1 Score: %.3f', f1_score), ...
         'FontSize', 11, 'FontWeight', 'bold', 'Color', 'blue');
    text(0.3, 0.47, sprintf('Precision: %.3f', precision), ...
         'FontSize', 11, 'Color', 'green');
    text(0.7, 0.47, sprintf('Recall: %.3f', recall), ...
         'FontSize', 11, 'Color', 'green');
    
    % 分类结果
    text(0.1, 0.30, 'Classification Results:', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.25, sprintf('Correctly Classified: %d (%.1f%%)', TP+TN, accuracy*100), ...
         'FontSize', 11, 'Color', 'green');
    text(0.1, 0.20, sprintf('Misclassified: %d (%.1f%%)', FP+FN, (1-accuracy)*100), ...
         'FontSize', 11, 'Color', 'red');
    
    % 模型建议
    if accuracy >= 0.85
        status_color = 'green';
        status_text = 'Excellent - Model performance is good';
    elseif accuracy >= 0.75
        status_color = 'blue';
        status_text = 'Good - Model is usable';
    elseif accuracy >= 0.65
        status_color = 'orange';
        status_text = 'Fair - Recommend adding training data';
    else
        status_color = 'red';
        status_text = 'Poor - Need to optimize features or data';
    end
    
    text(0.5, 0.10, status_text, 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', status_color);
    
    % 添加总标题
    sgtitle('Shanghai Dialect Detection Model Training Analysis Report', 'FontSize', 18, 'FontWeight', 'bold', ...
            'Color', [0, 0.3, 0.5]);
    
    % 添加训练时间戳
    annotation('textbox', [0.02, 0.02, 0.3, 0.03], 'String', ...
               sprintf('Training time: %s', datestr(now)), ...
               'EdgeColor', 'none', 'FontSize', 9, 'Color', [0.5, 0.5, 0.5]);
end