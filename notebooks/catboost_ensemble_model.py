# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, CatBoostRegressor
from IPython.core.interactiveshell import InteractiveShell
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.model_selection import train_test_split

# Setting styles
InteractiveShell.ast_node_interactivity = "all"
sns.set(style="whitegrid", color_codes=True)

random_state = 123

# %%

df = pd.read_csv(os.path.join("data", "processed", "train.csv"))
df = df.drop(columns=["train"])

# %%

X = df.drop(columns=["longitude", "latitude", "buildingid", "floor"])
y_lon = df.longitude
y_lat = df.latitude
y_building = df.buildingid
y_floor = df.floor

# %%

X_train, X_test, y_train_lon, y_test_lon = train_test_split(
    X,
    y_lon,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

_, _, y_train_lat, y_test_lat = train_test_split(
    X,
    y_lat,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

_, _, y_train_building, y_test_building = train_test_split(
    X,
    y_building,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)

_, _, y_train_floor, y_test_floor = train_test_split(
    X,
    y_floor,
    test_size=0.2,
    random_state=random_state,
    stratify=df[["longitude", "latitude", "buildingid", "floor"]],
)


# %%

catboost_lon = CatBoostRegressor(
    iterations=None,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="RMSE",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    best_model_min_trees=None,
    verbose=None,
    silent=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_metric=None,
    eval_metric="RMSE",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    pinned_memory_size=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    ctr_target_border_count=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)


catboost_lat = CatBoostRegressor(
    iterations=None,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="RMSE",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    best_model_min_trees=None,
    verbose=None,
    silent=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_metric=None,
    eval_metric="RMSE",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    pinned_memory_size=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    ctr_target_border_count=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)

catboost_building = CatBoostClassifier(
    iterations=None,
    learning_rate=0.2,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="MultiClass",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    verbose=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    classes_count=None,
    class_weights=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_loss=None,
    custom_metric=None,
    eval_metric="MultiClass",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    scale_pos_weight=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)

catboost_floor = CatBoostClassifier(
    iterations=None,
    learning_rate=0.2,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="MultiClass",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    verbose=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    classes_count=None,
    class_weights=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_loss=None,
    custom_metric=None,
    eval_metric="MultiClass",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    scale_pos_weight=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)

# %% Longitude

catboost_lon.fit(X_train, y_train_lon, eval_set=(X_test, y_test_lon))

# %% Latitude

catboost_lat.fit(X_train, y_train_lat, eval_set=(X_test, y_test_lat))

# %% Buildings

catboost_building.fit(X_train, y_train_building, eval_set=(X_test, y_test_building))

# %% Floors

catboost_floor.fit(X_train, y_train_floor, eval_set=(X_test, y_test_floor))

# %% Predicting with models

pred_train_lon = catboost_lon.predict(X_train)
pred_train_lat = catboost_lat.predict(X_train)
pred_train_buildings = catboost_building.predict_proba(X_train)
pred_train_floors = catboost_floor.predict_proba(X_train)

pred_test_lon = catboost_lon.predict(X_test)
pred_test_lat = catboost_lat.predict(X_test)
pred_test_buildings = catboost_building.predict_proba(X_test)
pred_test_floors = catboost_floor.predict_proba(X_test)

# %% untangling predictions for different classes

# not sure which predictions refers to which building and floor
pred_train_building0 = pred_train_buildings[:, 0]
pred_train_building1 = pred_train_buildings[:, 1]
pred_train_building2 = pred_train_buildings[:, 2]
pred_train_floor0 = pred_train_floors[:, 0]
pred_train_floor1 = pred_train_floors[:, 1]
pred_train_floor2 = pred_train_floors[:, 2]
pred_train_floor3 = pred_train_floors[:, 3]
pred_train_floor4 = pred_train_floors[:, 4]

# not sure which predictions refers to which building and floor
pred_test_building0 = pred_test_buildings[:, 0]
pred_test_building1 = pred_test_buildings[:, 1]
pred_test_building2 = pred_test_buildings[:, 2]
pred_test_floor0 = pred_test_floors[:, 0]
pred_test_floor1 = pred_test_floors[:, 1]
pred_test_floor2 = pred_test_floors[:, 2]
pred_test_floor3 = pred_test_floors[:, 3]
pred_test_floor4 = pred_test_floors[:, 4]

# %% Creating a new training set from the predictions

X_train_comb = pd.DataFrame(
    {
        "lon": pred_train_lon,
        "lat": pred_train_lat,
        "building0": pred_train_building0,
        "building1": pred_train_building1,
        "building2": pred_train_building2,
        "floor0": pred_train_floor0,
        "floor1": pred_train_floor1,
        "floor2": pred_train_floor2,
        "floor3": pred_train_floor3,
        "floor4": pred_train_floor4,
    }
)

X_test_comb = pd.DataFrame(
    {
        "lon": pred_test_lon,
        "lat": pred_test_lat,
        "building0": pred_test_building0,
        "building1": pred_test_building1,
        "building2": pred_test_building2,
        "floor0": pred_test_floor0,
        "floor1": pred_test_floor1,
        "floor2": pred_test_floor2,
        "floor3": pred_test_floor3,
        "floor4": pred_test_floor4,
    }
)


# %% New combination models

catboost_lon_comb = CatBoostRegressor(
    iterations=None,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="RMSE",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    best_model_min_trees=None,
    verbose=None,
    silent=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_metric=None,
    eval_metric="RMSE",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    pinned_memory_size=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    ctr_target_border_count=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)


catboost_lat_comb = CatBoostRegressor(
    iterations=None,
    learning_rate=None,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="RMSE",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    best_model_min_trees=None,
    verbose=None,
    silent=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_metric=None,
    eval_metric="RMSE",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    pinned_memory_size=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    ctr_target_border_count=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)

catboost_floor_comb = CatBoostClassifier(
    iterations=None,
    learning_rate=0.2,
    depth=None,
    l2_leaf_reg=None,
    model_size_reg=None,
    rsm=None,
    loss_function="MultiClass",
    border_count=None,
    feature_border_type=None,
    per_float_feature_quantization=None,
    input_borders=None,
    output_borders=None,
    fold_permutation_block=None,
    od_pval=None,
    od_wait=None,
    od_type=None,
    nan_mode=None,
    counter_calc_method=None,
    leaf_estimation_iterations=None,
    leaf_estimation_method=None,
    thread_count=None,
    use_best_model=True,
    verbose=None,
    logging_level=None,
    metric_period=None,
    ctr_leaf_count_limit=None,
    store_all_simple_ctr=None,
    max_ctr_complexity=None,
    has_time=None,
    allow_const_label=None,
    classes_count=None,
    class_weights=None,
    one_hot_max_size=None,
    random_strength=None,
    name=None,
    ignored_features=None,
    train_dir=None,
    custom_loss=None,
    custom_metric=None,
    eval_metric="MultiClass",
    bagging_temperature=None,
    save_snapshot=None,
    snapshot_file=None,
    snapshot_interval=None,
    fold_len_multiplier=None,
    used_ram_limit=None,
    gpu_ram_part=None,
    allow_writing_files=None,
    final_ctr_computation_mode=None,
    approx_on_full_history=None,
    boosting_type=None,
    simple_ctr=None,
    combinations_ctr=None,
    per_feature_ctr=None,
    task_type=None,
    device_config=None,
    devices=None,
    bootstrap_type=None,
    subsample=None,
    sampling_unit=None,
    dev_score_calc_obj_block_size=None,
    max_depth=None,
    n_estimators=None,
    num_boost_round=None,
    num_trees=None,
    colsample_bylevel=None,
    random_state=random_state,
    reg_lambda=None,
    objective=None,
    eta=None,
    max_bin=None,
    scale_pos_weight=None,
    gpu_cat_features_storage=None,
    data_partition=None,
    metadata=None,
    early_stopping_rounds=30,
    cat_features=None,
    grow_policy=None,
    min_data_in_leaf=None,
    min_child_samples=None,
    max_leaves=None,
    num_leaves=None,
    score_function=None,
    leaf_estimation_backtracking=None,
    ctr_history_unit=None,
    monotone_constraints=None,
)


# %% Longitude combination model

catboost_lon_comb.fit(X_train_comb, y_train_lon, eval_set=(X_test_comb, y_test_lon))

# %% Latitude combination model

catboost_lat_comb.fit(X_train_comb, y_train_lat, eval_set=(X_test_comb, y_test_lat))

# %% Floor combination model

catboost_floor_comb.fit(
    X_train_comb, y_train_floor, eval_set=(X_test_comb, y_test_floor)
)

# %% Predicting with combination models

pred_test_lon_comb = catboost_lon_comb.predict(X_test_comb)
pred_test_lat_comb = catboost_lat_comb.predict(X_test_comb)
pred_test_floor_comb = catboost_floor_comb.predict(X_test_comb)

predictions = pd.DataFrame(
    {
        "LATITUDE": pred_test_lat_comb,
        "LONGITUDE": pred_test_lon_comb,
        "FLOOR": np.hstack(pred_test_floor_comb),
    }
)

true_values = pd.DataFrame(
    {"LATITUDE": y_test_lat, "LONGITUDE": y_test_lon, "FLOOR": y_test_floor}
)


# %%

for floor in sorted(predictions.FLOOR.unique()):
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="LONGITUDE",
        y="LATITUDE",
        data=predictions[predictions["FLOOR"] == int(floor)],
        ax=ax,
    )
    ax.set_aspect(aspect="equal")
    plt.title(f"Predictions Floor {int(floor)}")
    plt.show()

    fig, ax = plt.subplots()
    sns.scatterplot(
        x="LONGITUDE",
        y="LATITUDE",
        data=true_values[true_values["FLOOR"] == int(floor)],
        ax=ax,
    )
    ax.set_aspect(aspect="equal")
    plt.title(f"Real Values Floor {int(floor)}")
    plt.show()


# %%
