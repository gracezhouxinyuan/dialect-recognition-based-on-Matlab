function shanghai_dialect_recognition_demo()
    % 初始化参数
    params = init_parameters();
    
    % 加载现有模型（如果存在）
    params = load_existing_model(params);
    
    while true
        fprintf('\n=== 沪语检测系统 ===\n');
        fprintf('1. 检测音频文件\n');
        fprintf('2. 训练模型\n');
        fprintf('3. 快速训练（10+10样本）\n');
        fprintf('4. 更新检测参数\n');
        fprintf('5. 显示当前参数\n');
        fprintf('6. 退出\n');
        
        choice = input('请选择 (1-6): ', 's');
        
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
                fprintf('退出系统\n');
                break;
            otherwise
                fprintf('无效选择\n');
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
    params.training_stats.last_trained = '从未训练';
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
                fprintf('发现已保存的模型，正在加载...\n');
                loaded_params = loaded_data.params;
                
                % 合并参数，保留新版本的新增字段
                params = merge_parameters(params, loaded_params);
                
                params.training_stats.last_trained = '从保存文件加载';
                fprintf('模型加载成功！\n');
            end
        catch ME
            fprintf('加载模型失败: %s\n', ME.message);
        end
    end
    
    % 加载快速训练模型（只有在字段存在时才执行）
    if isfield(params, 'quick_train_file') && exist(params.quick_train_file, 'file')
        try
            loaded_data = load(params.quick_train_file);
            if isfield(loaded_data, 'quick_params')
                fprintf('发现快速训练参数，正在加载...\n');
                params.quick_params = loaded_data.quick_params;
                fprintf('快速训练参数加载成功！\n');
            end
        catch ME
            fprintf('加载快速训练参数失败: %s\n', ME.message);
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
        new_params.training_stats.last_trained = '从旧版本升级';
        new_params.training_stats.samples_used = 0;
        new_params.training_stats.accuracy = 0;
    end
end
%%
function params = quick_train_model(params)
    % 快速训练模型 - 使用10组沪语和10组非沪语样本
    fprintf('\n=== 快速模型训练 (10+10样本) ===\n');
    
    % 选择训练数据文件夹
    train_dir = uigetdir('', '选择训练数据文件夹（应包含shanghai和non_shanghai子文件夹）');
    if isequal(train_dir, 0)
        fprintf('未选择训练文件夹\n');
        return;
    end
    
    % 检查文件夹结构
    shanghai_dir = fullfile(train_dir, 'shanghai');
    non_shanghai_dir = fullfile(train_dir, 'non_shanghai');
    
    fprintf('检查目录:\n');
    fprintf('  主目录: %s\n', train_dir);
    fprintf('  沪语目录: %s - 存在: %d\n', shanghai_dir, exist(shanghai_dir, 'dir'));
    fprintf('  非沪语目录: %s - 存在: %d\n', non_shanghai_dir, exist(non_shanghai_dir, 'dir'));
    
    if ~exist(shanghai_dir, 'dir') || ~exist(non_shanghai_dir, 'dir')
        fprintf('错误：训练文件夹应包含shanghai和non_shanghai子文件夹\n');
        return;
    end
    
    % 获取训练文件
    shanghai_files = dir(fullfile(shanghai_dir, '*.wav'));
    non_shanghai_files = dir(fullfile(non_shanghai_dir, '*.wav'));
    
    fprintf('找到文件:\n');
    fprintf('  沪语文件: %d 个\n', length(shanghai_files));
    fprintf('  非沪语文件: %d 个\n', length(non_shanghai_files));
    
    if isempty(shanghai_files) || isempty(non_shanghai_files)
        fprintf('错误：未找到足够的训练文件\n');
        return;
    end
    
    % 限制每个类别最多10个样本
    num_shanghai = min(10, length(shanghai_files));
    num_non_shanghai = min(10, length(non_shanghai_files));
    
    fprintf('使用 %d 个沪语样本和 %d 个非沪语样本进行快速训练\n', num_shanghai, num_non_shanghai);
    
    % 提取特征和标签
    features = [];
    labels = [];
    shanghai_features = [];
    non_shanghai_features = [];
    
    % 处理沪语样本
    fprintf('处理沪语样本...\n');
    for i = 1:num_shanghai
        filename = fullfile(shanghai_dir, shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                shanghai_features = [shanghai_features; audio_features];
                labels = [labels; 1];
                fprintf('沪语样本 %d/%d: %s\n', i, num_shanghai, shanghai_files(i).name);
            end
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', shanghai_files(i).name, ME.message);
        end
    end
    
    % 处理非沪语样本
    fprintf('处理非沪语样本...\n');
    for i = 1:num_non_shanghai
        filename = fullfile(non_shanghai_dir, non_shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                non_shanghai_features = [non_shanghai_features; audio_features];
                labels = [labels; 0];
                fprintf('非沪语样本 %d/%d: %s\n', i, num_non_shanghai, non_shanghai_files(i).name);
            end
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', non_shanghai_files(i).name, ME.message);
        end
    end
    
    if isempty(features) || size(shanghai_features, 1) == 0 || size(non_shanghai_features, 1) == 0
        fprintf('错误：未能提取到足够的有效特征\n');
        return;
    end
    
    fprintf('特征提取完成，共 %d 个样本\n', size(features, 1));
    
    % 计算特征统计信息
    quick_params = calculate_quick_parameters(shanghai_features, non_shanghai_features);
    
    % 保存快速训练参数
    params.quick_params = quick_params;
    save(params.quick_train_file, 'quick_params');
    
    % 更新训练统计
    params.training_stats.last_trained = datestr(now);
    params.training_stats.samples_used = size(features, 1);
    params.training_stats.quick_training = true;
    
    fprintf('快速训练完成！参数已保存到: %s\n', params.quick_train_file);
    fprintf('可以使用选项4来应用这些参数到检测系统\n');
    
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
    
    fprintf('基于训练数据计算出的特征权重:\n');
    feature_names = {'低频能量', '基频变化', '过零率', '能量变化'};
    for i = 1:4
        fprintf('  %s: %.3f (分离度: %.3f)\n', feature_names{i}, ...
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
    fprintf('\n=== 快速训练结果摘要 ===\n');
    fprintf('沪语样本数: %d\n', size(shanghai_features, 1));
    fprintf('非沪语样本数: %d\n', size(non_shanghai_features, 1));
    
    feature_names = {'低频能量', '基频变化', '过零率', '能量变化'};
    
    fprintf('\n特征统计对比:\n');
    fprintf('%-12s %-10s %-10s %-10s\n', '特征', '沪语均值', '非沪语均值', '分离度');
    fprintf('%-12s %-10s %-10s %-10s\n', '------', '--------', '----------', '------');
    
    for i = 1:4
        feature_name = get_feature_name(i);
        shanghai_mean = quick_params.(['shanghai_mean_', feature_name]);
        non_shanghai_mean = quick_params.(['non_shanghai_mean_', feature_name]);
        separation = quick_params.(['separation_', feature_name]);
        
        fprintf('%-12s %-10.3f %-10.3f %-10.3f\n', ...
                feature_names{i}, shanghai_mean, non_shanghai_mean, separation);
    end
    
    fprintf('\n优化的权重参数:\n');
    total_weight = 0;
    for i = 1:4
        weight = quick_params.optimized_weights(i);
        total_weight = total_weight + weight;
        fprintf('  %s: %.3f\n', feature_names{i}, weight);
    end
    fprintf('总权重: %.3f\n', total_weight);
end
%%
function params = update_detection_parameters(params)
    % 更新检测参数
    fprintf('\n=== 更新检测参数 ===\n');
    
    if ~isfield(params, 'quick_params')
        fprintf('错误：未找到训练参数，请先进行快速训练\n');
        return;
    end
    
    quick_params = params.quick_params;
    
    % 显示当前参数和推荐参数
    fprintf('当前参数 vs 推荐参数:\n');
    fprintf('%-15s %-8s %-8s\n', '参数', '当前', '推荐');
    fprintf('%-15s %-8s %-8s\n', '----', '----', '----');
    
    current_weights = [params.low_freq_weight, params.pitch_var_weight, ...
                      params.zcr_weight, params.energy_var_weight];
    
    feature_names = {'低频权重', '基频权重', '过零率权重', '能量权重'};
    
    for i = 1:4
        fprintf('%-15s %-8.3f %-8.3f\n', feature_names{i}, ...
                current_weights(i), quick_params.optimized_weights(i));
    end
    
    fprintf('%-15s %-8.3f %-8.3f\n', '阈值', params.shanghai_threshold, quick_params.optimized_threshold);
    
    % 询问用户是否更新
    choice = input('是否应用推荐参数？(y/n): ', 's');
    
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
        
        fprintf('参数更新成功！\n');
        fprintf('新的权重: [%.3f, %.3f, %.3f, %.3f]\n', ...
                params.low_freq_weight, params.pitch_var_weight, ...
                params.zcr_weight, params.energy_var_weight);
        fprintf('新的阈值: %.3f\n', params.shanghai_threshold);
        
        % 更新训练统计
        params.training_stats.parameters_updated = datestr(now);
        params.training_stats.using_optimized_params = true;
    else
        fprintf('参数未更新，继续使用当前参数\n');
    end
end
%%
function display_current_parameters(params)
    % 显示当前参数
    fprintf('\n=== 当前检测参数 ===\n');
    
    fprintf('基本参数:\n');
    fprintf('  采样率: %d Hz\n', params.fs);
    fprintf('  帧长: %.3f s\n', params.frame_len);
    fprintf('  帧移: %.3f s\n', params.hop_len);
    fprintf('  最小音频时长: %.1f s\n', params.min_duration);
    
    fprintf('\n检测阈值和权重:\n');
    fprintf('  沪语判断阈值: %.3f\n', params.shanghai_threshold);
    fprintf('  低频能量权重: %.3f\n', params.low_freq_weight);
    fprintf('  基频变化权重: %.3f\n', params.pitch_var_weight);
    fprintf('  过零率权重: %.3f\n', params.zcr_weight);
    fprintf('  能量变化权重: %.3f\n', params.energy_var_weight);
    
    if isfield(params, 'training_stats')
        fprintf('\n训练统计:\n');
        fprintf('  最后训练时间: %s\n', params.training_stats.last_trained);
        fprintf('  使用样本数: %d\n', params.training_stats.samples_used);
        if isfield(params.training_stats, 'using_optimized_params') && params.training_stats.using_optimized_params
            fprintf('  当前使用优化参数: 是\n');
        else
            fprintf('  当前使用优化参数: 否\n');
        end
    end
    
    if isfield(params, 'quick_params')
        fprintf('\n快速训练参数可用: 是\n');
        fprintf('  使用选项4来应用这些参数\n');
    else
        fprintf('\n快速训练参数可用: 否\n');
        fprintf('  使用选项3来进行快速训练\n');
    end
end
%%
function train_model(params)
    % 训练沪语检测模型
    fprintf('\n=== 沪语检测模型训练 ===\n');
    
    % 选择训练数据文件夹
    train_dir = uigetdir('', '选择训练数据文件夹（应包含shanghai和non_shanghai子文件夹）');
    if isequal(train_dir, 0)
        fprintf('未选择训练文件夹\n');
        return;
    end
    
    % 检查文件夹结构
    shanghai_dir = fullfile(train_dir, 'shanghai');
    non_shanghai_dir = fullfile(train_dir, 'non_shanghai');
    
    if ~exist(shanghai_dir, 'dir') || ~exist(non_shanghai_dir, 'dir')
        fprintf('错误：训练文件夹应包含shanghai和non_shanghai子文件夹\n');
        return;
    end
    
    % 获取训练文件
    shanghai_files = dir(fullfile(shanghai_dir, '*.wav'));
    non_shanghai_files = dir(fullfile(non_shanghai_dir, '*.wav'));
    
    if isempty(shanghai_files) && isempty(non_shanghai_files)
        fprintf('错误：未找到WAV格式的训练文件\n');
        return;
    end
    
    fprintf('找到 %d 个沪语样本和 %d 个非沪语样本\n', ...
            length(shanghai_files), length(non_shanghai_files));
    
    % 提取特征和标签
    features = [];
    labels = [];
    
    % 处理沪语样本
    fprintf('处理沪语样本...\n');
    for i = 1:length(shanghai_files)
        filename = fullfile(shanghai_dir, shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                labels = [labels; 1];  % 沪语标签为1
                fprintf('处理沪语样本 %d/%d: %s\n', i, length(shanghai_files), shanghai_files(i).name);
            end
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', shanghai_files(i).name, ME.message);
        end
    end
    
    % 处理非沪语样本
    fprintf('处理非沪语样本...\n');
    for i = 1:length(non_shanghai_files)
        filename = fullfile(non_shanghai_dir, non_shanghai_files(i).name);
        try
            audio_features = extract_features_from_file(filename, params);
            if ~isempty(audio_features)
                features = [features; audio_features];
                labels = [labels; 0];  % 非沪语标签为0
                fprintf('处理非沪语样本 %d/%d: %s\n', i, length(non_shanghai_files), non_shanghai_files(i).name);
            end
        catch ME
            fprintf('处理文件 %s 时出错: %s\n', non_shanghai_files(i).name, ME.message);
        end
    end
    
    if isempty(features)
        fprintf('错误：未能提取到任何有效特征\n');
        return;
    end
    
    fprintf('特征提取完成，共 %d 个样本\n', size(features, 1));
    
    % 训练分类器
    fprintf('训练分类器...\n');
    model = train_classifier(features, labels);
    
    % 保存模型
    save(params.model_file, 'model', 'params');
    fprintf('模型已保存到: %s\n', params.model_file);
    
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
            fprintf('音频太短，跳过: %s\n', filename);
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
                    fprintf('警告: 特征字段 %s 缺失，使用默认值\n', required_fields{i});
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
                fprintf('警告: 文件 %s 包含无效特征，跳过\n', filename);
                features = [];
            end
        else
            fprintf('警告: 文件 %s 特征提取失败\n', filename);
            features = [];
        end
        
    catch ME
        fprintf('处理文件 %s 时出错: %s\n', filename, ME.message);
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
        fprintf('使用线性分类器失败，使用阈值分类器\n');
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
    [file, path] = uigetfile({'*.wav;*.mp3', '音频文件'}, '选择要检测的音频文件');
    if isequal(file, 0)
        fprintf('未选择文件\n');
        return;
    end
    
    filename = fullfile(path, file);
    fprintf('正在检测: %s\n', file);
    
    try
        % 读取音频
        [audio, fs] = audioread(filename);
        if size(audio, 2) > 1
            audio = mean(audio, 2); % 转为单声道
        end
        
        duration = length(audio) / fs;
        fprintf('音频信息: %.2f秒, %d Hz\n', duration, fs);
        
        % 检查音频长度
        if duration < params.min_duration
            fprintf('音频太短，至少需要%.1f秒\n', params.min_duration);
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
        fprintf('检测过程中出错: %s\n', ME.message);
    end
end
%%
function features = extract_shanghai_features(audio, fs, params)
    % 提取沪语关键特征
    fprintf('提取沪语特征...\n');
    
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
    
    fprintf('特征提取完成\n');
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
    
    % 1. 低频能量得分（沪语低频能量较高）
   % scores(1) = min(features.low_freq_ratio * 3, 1.0);
    
    % 2. 基频变化得分（沪语音调变化较大）
    %scores(2) = min(features.pitch_variance / 60, 1.0);
    
    % 3. 过零率得分（沪语辅音特征）
    %scores(3) = max(0, 1 - features.zcr / 0.12);
    
    % 4. 能量变化得分
    %scores(4) = min(features.energy_variance * 8, 1.0);
    
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
    
    fprintf('检测得分: %.3f, 阈值: %.3f\n', total_score, params.shanghai_threshold);
end
%%
function display_detection_result(is_shanghai, confidence, score, features, audio, fs)
    % 显示检测结果
    fprintf('\n=== 检测结果 ===\n');
    
    if is_shanghai
        fprintf('检测结果: 上海话\n');
        fprintf('置信度: %.1f%%\n', confidence * 100);
    else
        fprintf('检测结果: 其他语言\n');
        fprintf('置信度: %.1f%%\n', confidence * 100);
    end
    
    fprintf('综合得分: %.3f/1.000\n', score);
    
    fprintf('\n=== 特征详情 ===\n');
    fprintf('低频能量比: %.3f (沪语>0.3)\n', features.low_freq_ratio);
    fprintf('平均基频: %.1f Hz\n', features.pitch_mean);
    fprintf('基频变化: %.1f Hz (沪语>25)\n', features.pitch_variance);
    fprintf('过零率: %.3f (沪语<0.1)\n', features.zcr);
    fprintf('能量变化: %.3f\n', features.energy_variance);
    
    % 绘制结果图表
    plot_detection_results(is_shanghai, score, features, audio, fs);
end
%%
function plot_detection_results(is_shanghai, score, features, audio, fs)
    % 绘制检测结果图表 - 使用更简单的布局确保所有内容都显示
    figure('Name', '沪语检测分析', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 900]);
    
    % 使用2x2布局，给结果留出更大空间
    % 1. 音频波形
    subplot(2, 2, 1);
    t = (0:length(audio)-1) / fs;
    plot(t, audio, 'b', 'LineWidth', 1);
    title('音频波形', 'FontSize', 12);
    xlabel('时间 (s)'); ylabel('幅度'); grid on;
    
    % 2. 频谱分析
    subplot(2, 2, 2);
    N = min(1024, length(audio));
    Y = fft(audio(1:N), N);
    f = (0:N-1) * fs / N;
    plot(f(1:N/2), abs(Y(1:N/2)), 'r', 'LineWidth', 1);
    title('频谱分析', 'FontSize', 12);
    xlabel('频率 (Hz)'); ylabel('幅度'); grid on;
    
    % 标记沪语特征频率区域
    hold on;
    yl = ylim;
    fill([80, 400, 400, 80], [yl(1), yl(1), yl(2), yl(2)], 'g', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    legend('频谱', '沪语特征频带', 'Location', 'northeast');
    
    % 3. 特征条形图
    subplot(2, 2, 3);
    feature_names = {'低频能量', '基频变化', '过零率', '能量变化'};
    feature_values = [features.low_freq_ratio, features.pitch_variance/60, ...
                     1 - min(features.zcr/0.15, 1), ...
                     min(features.energy_variance*10, 1)];
    
    bar(feature_values, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', feature_names, 'FontSize', 10);
    title('归一化特征值', 'FontSize', 12); 
    ylabel('数值'); 
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
    text(0.5, 0.95, '沪语检测最终结果', 'FontSize', 20, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', [0, 0.4, 0.6]);
    
    % 分隔线
    line([0.1, 0.9], [0.90, 0.90], 'Color', 'k', 'LineWidth', 2);
    
    % 显著显示检测结果
    if is_shanghai
        % 上海话结果 - 红色突出显示
        text(0.5, 0.82, '上海话', 'FontSize', 28, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'red');
        text(0.5, 0.75, '检测到沪语方言', 'FontSize', 18, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'red');
    else
        % 其他语言结果 - 蓝色显示
        text(0.5, 0.82, '其他语言', 'FontSize', 28, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'blue');
        text(0.5, 0.75, '未检测到沪语特征', 'FontSize', 18, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'Color', 'blue');
    end
    
    % 置信度显示 - 位置上移
    text(0.5, 0.65, sprintf('置信度: %.1f%%', score * 100), 'FontSize', 20, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', result_color);
    
    % 得分信息框
    rectangle('Position', [0.2, 0.52, 0.6, 0.08], 'FaceColor', [1, 1, 1], ...
              'EdgeColor', 'k', 'LineWidth', 2);
    text(0.5, 0.57, sprintf('综合得分: %.3f', score), 'FontSize', 16, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    text(0.5, 0.54, sprintf('阈值: 0.600'), 'FontSize', 14, ...
         'HorizontalAlignment', 'center');
    
    % 判断状态
    if is_shanghai
        status_text = '✓ 超过阈值，判断为沪语';
        status_color = 'green';
    else
        status_text = '✗ 未达阈值，判断为其他语言';
        status_color = 'red';
    end
    text(0.5, 0.42, status_text, 'FontSize', 14, 'HorizontalAlignment', 'center', ...
         'Color', status_color, 'FontWeight', 'bold');
    
    % 特征匹配情况
    text(0.5, 0.35, '特征匹配情况:', 'FontSize', 12, 'HorizontalAlignment', 'center', ...
         'FontWeight', 'bold');
    
    % 显示各个特征的匹配状态
    feature_status = {
        sprintf('低频能量: %.3f %s', features.low_freq_ratio, iif(features.low_freq_ratio > 0.3, '✓', '✗')),
        sprintf('基频变化: %.1f Hz %s', features.pitch_variance, iif(features.pitch_variance > 25, '✓', '✗')),
        sprintf('过零率: %.3f %s', features.zcr, iif(features.zcr < 0.12, '✓', '✗'))
    };
    
    for i = 1:length(feature_status)
        text(0.5, 0.30 - (i-1)*0.05, feature_status{i}, 'FontSize', 10, ...
             'HorizontalAlignment', 'center');
    end
    
    % 音频基本信息
    text(0.5, 0.05, sprintf('音频时长: %.1f秒 | 采样率: %d Hz', length(audio)/fs, fs), ...
         'FontSize', 10, 'HorizontalAlignment', 'center', 'Color', [0.5, 0.5, 0.5]);
    
    % 在整个图的上方添加总标题
    sgtitle(sprintf('沪语检测分析 - 最终结果: %s (得分: %.3f)', ...
                   iif(is_shanghai, '上海话', '其他语言'), score), ...
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
    fprintf('\n=== 训练结果 ===\n');
    
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
    fprintf('训练集准确率: %.2f%%\n', accuracy * 100);
    
    % 计算混淆矩阵
    TP = sum(predictions == 1 & labels == 1);
    FP = sum(predictions == 1 & labels == 0);
    TN = sum(predictions == 0 & labels == 0);
    FN = sum(predictions == 0 & labels == 1);
    
    fprintf('混淆矩阵:\n');
    fprintf('          预测沪语   预测非沪语\n');
    fprintf('实际沪语    %3d        %3d\n', TP, FN);
    fprintf('实际非沪语  %3d        %3d\n', FP, TN);
    
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
    
    fprintf('\n详细指标:\n');
    fprintf('精确率: %.3f\n', precision);
    fprintf('召回率: %.3f\n', recall);
    fprintf('F1分数: %.3f\n', f1_score);
    
    % 显示特征重要性
    fprintf('\n特征重要性:\n');
    feature_names = {'低频能量', '基频变化', '过零率', '能量变化'};
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
    
    figure('Name', '沪语检测模型训练结果', 'NumberTitle', 'off', ...
           'Position', [100, 50, 1400, 900]);
    
    % 1. 性能指标雷达图
    subplot(2, 3, 1);
    performance_metrics = [accuracy, precision, recall, f1_score];
    metric_names = {'准确率', '精确率', '召回率', 'F1分数'};
    
    % 创建雷达图
    angles = linspace(0, 2*pi, length(performance_metrics) + 1);
    angles = angles(1:end-1);
    
    polarplot([angles, angles(1)], [performance_metrics, performance_metrics(1)], ...
              'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 8);
    thetaticks(angles * 180/pi);
    thetaticklabels(metric_names);
    rlim([0 1]);
    title('性能指标雷达图', 'FontSize', 12, 'FontWeight', 'bold');
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
    set(gca, 'XTick', 1:2, 'XTickLabel', {'预测沪语', '预测非沪语'});
    set(gca, 'YTick', 1:2, 'YTickLabel', {'实际沪语', '实际非沪语'});
    set(gca, 'XTickLabelRotation', 45);
    
    % 添加数值文本
    for i = 1:2
        for j = 1:2
            text(j, i, sprintf('%d\n(%.1f%%)', confusion_mat(i,j), heatmap_data(i,j)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                 'FontWeight', 'bold', 'Color', 'white', 'FontSize', 11);
        end
    end
    
    title('混淆矩阵热图 (%)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 3. 特征重要性条形图
    subplot(2, 3, 3);
    feature_names = {'低频能量', '基频变化', '过零率', '能量变化'};
    
    barh(model.feature_importance, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'black');
    set(gca, 'YTickLabel', feature_names);
    xlabel('重要性得分');
    title('特征重要性', 'FontSize', 12, 'FontWeight', 'bold');
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
            legend('非沪语', '沪语', 'Location', 'best');
            title('特征空间分布（前2个主要特征）', 'FontSize', 12, 'FontWeight', 'bold');
            grid on;
        else
            text(0.5, 0.5, '特征数量不足', 'HorizontalAlignment', 'center', ...
                 'FontSize', 12, 'FontWeight', 'bold');
            title('特征空间分布', 'FontSize', 12, 'FontWeight', 'bold');
            axis off;
        end
    else
        text(0.5, 0.5, '特征维度不足', 'HorizontalAlignment', 'center', ...
             'FontSize', 12, 'FontWeight', 'bold');
        title('特征空间分布', 'FontSize', 12, 'FontWeight', 'bold');
        axis off;
    end
    
    % 5. 分数分布直方图
    subplot(2, 3, 5);
    histogram(confidence_scores(labels==1), 'FaceColor', 'red', 'FaceAlpha', 0.6, 'BinWidth', 0.1);
    hold on;
    histogram(confidence_scores(labels==0), 'FaceColor', 'blue', 'FaceAlpha', 0.6, 'BinWidth', 0.1);
    xlabel('置信度分数');
    ylabel('频数');
    title('置信度分数分布', 'FontSize', 12, 'FontWeight', 'bold');
    legend('沪语', '非沪语');
    grid on;
    
    % 6. 训练摘要面板
    subplot(2, 3, 6);
    axis off;
    
    % 背景设置
    set(gca, 'Color', [0.95, 0.95, 0.95]);
    
    % 标题
    text(0.5, 0.95, '训练结果摘要', 'FontSize', 16, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', [0, 0.4, 0.6]);
    
    % 分隔线
    line([0.1, 0.9], [0.90, 0.90], 'Color', 'k', 'LineWidth', 2);
    
    % 关键指标
    text(0.1, 0.80, sprintf('模型类型: %s', model.model_type), ...
         'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.75, sprintf('总样本数: %d', length(labels)), ...
         'FontSize', 11);
    text(0.1, 0.70, sprintf('沪语样本: %d', sum(labels==1)), ...
         'FontSize', 11);
    text(0.1, 0.65, sprintf('非沪语样本: %d', sum(labels==0)), ...
         'FontSize', 11);
    
    % 性能指标框
    rectangle('Position', [0.1, 0.45, 0.8, 0.15], 'FaceColor', 'white', ...
              'EdgeColor', 'k', 'LineWidth', 2);
    
    text(0.5, 0.57, '性能指标', 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center');
    
    text(0.3, 0.52, sprintf('准确率: %.1f%%', accuracy*100), ...
         'FontSize', 11, 'FontWeight', 'bold', 'Color', 'blue');
    text(0.7, 0.52, sprintf('F1分数: %.3f', f1_score), ...
         'FontSize', 11, 'FontWeight', 'bold', 'Color', 'blue');
    text(0.3, 0.47, sprintf('精确率: %.3f', precision), ...
         'FontSize', 11, 'Color', 'green');
    text(0.7, 0.47, sprintf('召回率: %.3f', recall), ...
         'FontSize', 11, 'Color', 'green');
    
    % 分类结果
    text(0.1, 0.30, '分类结果:', 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.25, sprintf('正确分类: %d (%.1f%%)', TP+TN, accuracy*100), ...
         'FontSize', 11, 'Color', 'green');
    text(0.1, 0.20, sprintf('错误分类: %d (%.1f%%)', FP+FN, (1-accuracy)*100), ...
         'FontSize', 11, 'Color', 'red');
    
    % 模型建议
    if accuracy >= 0.85
        status_color = 'green';
        status_text = '优秀 - 模型性能良好';
    elseif accuracy >= 0.75
        status_color = 'blue';
        status_text = '良好 - 模型可用';
    elseif accuracy >= 0.65
        status_color = 'orange';
        status_text = '一般 - 建议增加训练数据';
    else
        status_color = 'red';
        status_text = '较差 - 需要优化特征或数据';
    end
    
    text(0.5, 0.10, status_text, 'FontSize', 12, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', status_color);
    
    % 添加总标题
    sgtitle('沪语检测模型训练分析报告', 'FontSize', 18, 'FontWeight', 'bold', ...
            'Color', [0, 0.3, 0.5]);
    
    % 添加训练时间戳
    annotation('textbox', [0.02, 0.02, 0.3, 0.03], 'String', ...
               sprintf('训练时间: %s', datestr(now)), ...
               'EdgeColor', 'none', 'FontSize', 9, 'Color', [0.5, 0.5, 0.5]);
end