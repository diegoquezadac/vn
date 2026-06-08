import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Literal
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge


def _get_feature_importance(model, num_columns, cat_columns):
    feature_importance = list(
        zip(num_columns + cat_columns, model.get_feature_importance())
    )
    sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    return dict(sorted_importance)


def train_catboost(
    data: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[CatBoostRegressor, Dict]:
    results = {}
    plot = config.get("plot")
    target = config.get("target", "price")
    num_columns = config.get("num_columns").copy()
    cat_columns = config.get("cat_columns").copy()

    df = data.copy(deep=True)
    df[target] = df[target].apply(np.log)
    df[cat_columns] = df[cat_columns].fillna("").astype(str).replace("nan", "")

    model = CatBoostRegressor(
        loss_function=config.get("loss_function", "RMSE"),
        monotone_constraints=config.get("monotone_constraints", None),
        eval_metric="MAPE",
        early_stopping_rounds=5,
    )

    if config.get("select_features"):
        # NOTE: Check a better methodology for this validation
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=21)

        train_pool = Pool(
            df_train[num_columns + cat_columns],
            df_train[target],
            cat_features=cat_columns,
        )

        val_pool = Pool(
            df_val[num_columns + cat_columns],
            df_val[target],
            cat_features=cat_columns,
        )

        summary = model.select_features(
            train_pool,
            eval_set=val_pool,
            features_for_select=num_columns + cat_columns,
            num_features_to_select=round(0.7 * (len(num_columns + cat_columns))),
            steps=5,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=False,
            logging_level="Silent",
            plot=False,
        )

        selected_features = summary["selected_features_names"]

        num_columns = [column for column in num_columns if column in selected_features]
        cat_columns = [column for column in cat_columns if column in selected_features]

        results["selected_features"] = selected_features

    if config.get("validation") == "holdout":
        df_train, df_val = train_test_split(
            df, test_size=config.get("val_size", 0.12), random_state=42
        )

        train_pool = Pool(
            df_train[num_columns + cat_columns],
            df_train[target],
            cat_features=cat_columns,
        )

        val_pool = Pool(
            df_val[num_columns + cat_columns],
            df_val[target],
            cat_features=cat_columns,
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            plot=plot,
            # plot_file=f"./figures/validation_{tag}" if tag else None,
            verbose=False,
        )

    elif config.get("validation") == "cv":
        train_pool = Pool(
            df[num_columns + cat_columns],
            df[target],
            cat_features=cat_columns,
        )

        if config.get("grid_search"):
            search_results = model.grid_search(
                config.get("grid_search"),
                train_pool,
                # cv=3,
                plot=plot,
                # plot_file=f"./figures/validation_{tag}" if tag else None,
                shuffle=True,
                verbose=False,
                refit=True,
                partition_random_seed=21,
            )
            results["best_params"] = search_results["params"]
        else:
            raise KeyError("grid_search config is missing for running cv")

    else:
        train_pool = Pool(
            df[num_columns + cat_columns],
            df[target],
            cat_features=cat_columns,
        )

        model.fit(train_pool, verbose=False, plot=plot)

    results["feature_importance"] = _get_feature_importance(
        model, num_columns, cat_columns
    )

    return model, results


def train_sklearn_model(
    model: Literal["xgboost", "lightgbm", "random_forest", "stacking"],
    data: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Pipeline, Dict]:
    results = {}
    plot = config.get("plot")
    target = config.get("target", "price")
    num_columns = config.get("num_columns").copy()
    cat_columns = config.get("cat_columns").copy()

    df = data.copy(deep=True)
    df[target] = df[target].apply(np.log)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_columns),
            (
                "cat",
                TargetEncoder(smooth="auto", target_type="continuous"),
                cat_columns,
            ),
        ]
    )

    if model == "xgboost":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mape",
            random_state=42,
        )
    elif model == "lightgbm":
        estimator = LGBMRegressor(
            objective="regression",
            metric="mape",
            random_state=42,
            verbosity=-1,
        )
    elif model == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=42
        )
    elif model == "stacking":
        stacking_cv = config.get("stacking_cv", 5)
        estimator = StackingRegressor(
            estimators=[
                (
                    "catboost",
                    CatBoostRegressor(
                        loss_function="RMSE", random_state=42, verbose=False
                    ),
                ),
                (
                    "xgb",
                    XGBRegressor(
                        objective="reg:squarederror", random_state=42, verbosity=0
                    ),
                ),
                (
                    "lgbm",
                    LGBMRegressor(
                        objective="regression", random_state=42, verbosity=-1
                    ),
                ),
            ],
            final_estimator=Ridge(),
            cv=stacking_cv,
        )
    else:
        raise ValueError(
            f"Unknown model: {model}. Choose from 'xgboost', 'lightgbm', 'random_forest', 'stacking'."
        )

    supports_eval_set = model in ("xgboost", "lightgbm")

    if config.get("validation") == "holdout":
        if supports_eval_set:
            estimator.set_params(early_stopping_rounds=5)

        df_train, df_val = train_test_split(
            df, test_size=config.get("val_size", 0.12), random_state=42
        )

        X_train = df_train[num_columns + cat_columns]
        y_train = df_train[target]
        X_val = df_val[num_columns + cat_columns]
        y_val = df_val[target]

        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_val_transformed = preprocessor.transform(X_val)

        fit_params = {}
        if supports_eval_set:
            fit_params["eval_set"] = [(X_val_transformed, y_val)]
        if model == "xgboost":
            fit_params["verbose"] = False

        estimator.fit(X_train_transformed, y_train, **fit_params)

        y_pred_train = estimator.predict(X_train_transformed)
        y_pred_val = estimator.predict(X_val_transformed)
        results["train_mape"] = np.mean(
            np.abs((np.exp(y_pred_train) - np.exp(y_train)) / np.exp(y_train))
        )
        results["val_mape"] = np.mean(
            np.abs((np.exp(y_pred_val) - np.exp(y_val)) / np.exp(y_val))
        )
        results["train_mae"] = np.mean(np.abs(np.exp(y_pred_train) - np.exp(y_train)))
        results["val_mae"] = np.mean(np.abs(np.exp(y_pred_val) - np.exp(y_val)))

    elif config.get("validation") == "cv":
        X = df[num_columns + cat_columns]
        y = df[target]

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        if config.get("grid_search"):
            from sklearn.model_selection import GridSearchCV

            grid_params = {
                f"model__{k}": v for k, v in config.get("grid_search").items()
            }

            grid_search = GridSearchCV(
                pipeline,
                grid_params,
                cv=3,
                scoring="neg_mean_absolute_percentage_error",
                verbose=0 if not plot else 1,
                n_jobs=-1,
            )
            grid_search.fit(X, y)
            best_pipeline = grid_search.best_estimator_
            results["best_params"] = grid_search.best_params_
            return best_pipeline, results
        else:
            raise KeyError("grid_search config is missing for running cv")

    else:
        X = df[num_columns + cat_columns]
        y = df[target]

        X_transformed = preprocessor.fit_transform(X, y)
        fit_params = {}
        if model == "xgboost":
            fit_params["verbose"] = False
        estimator.fit(X_transformed, y, **fit_params)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    if hasattr(estimator, "feature_importances_"):
        feature_importance = list(
            zip(num_columns + cat_columns, estimator.feature_importances_)
        )
        sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        results["feature_importance"] = dict(sorted_importance)
    elif model == "stacking":
        stacking_model = pipeline.named_steps["model"]
        results["estimator_weights"] = dict(
            zip(
                [name for name, _ in stacking_model.estimators],
                stacking_model.final_estimator_.coef_,
            )
        )

    return pipeline, results


def train_stacking(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Pipeline, Dict]:
    results = {}
    plot = config.get("plot")
    target = config.get("target", "price")
    num_columns = config.get("num_columns").copy()
    cat_columns = config.get("cat_columns").copy()

    df = data.copy(deep=True)
    df[target] = df[target].apply(np.log)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_columns),
            (
                "cat",
                TargetEncoder(smooth="auto", target_type="continuous"),
                cat_columns,
            ),
        ]
    )

    # Base estimators without early_stopping (stacking's internal CV doesn't provide eval_set)
    stacking_cv = config.get("stacking_cv", 5)
    estimators = [
        (
            "catboost",
            CatBoostRegressor(loss_function="RMSE", random_state=42, verbose=False),
        ),
        (
            "xgb",
            XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0),
        ),
        ("lgbm", LGBMRegressor(objective="regression", random_state=42, verbosity=-1)),
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=stacking_cv,
    )

    if config.get("validation") == "holdout":
        df_train, df_val = train_test_split(
            df, test_size=config.get("val_size", 0.12), random_state=42
        )

        X_train = df_train[num_columns + cat_columns]
        y_train = df_train[target]
        X_val = df_val[num_columns + cat_columns]
        y_val = df_val[target]

        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_val_transformed = preprocessor.transform(X_val)
        model.fit(X_train_transformed, y_train)

        y_pred_train = model.predict(X_train_transformed)
        y_pred_val = model.predict(X_val_transformed)
        results["train_mape"] = np.mean(
            np.abs((np.exp(y_pred_train) - np.exp(y_train)) / np.exp(y_train))
        )
        results["val_mape"] = np.mean(
            np.abs((np.exp(y_pred_val) - np.exp(y_val)) / np.exp(y_val))
        )
        results["train_mae"] = np.mean(np.abs(np.exp(y_pred_train) - np.exp(y_train)))
        results["val_mae"] = np.mean(np.abs(np.exp(y_pred_val) - np.exp(y_val)))

    elif config.get("validation") == "cv":
        X = df[num_columns + cat_columns]
        y = df[target]

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        if config.get("grid_search"):
            from sklearn.model_selection import GridSearchCV

            grid_params = {
                f"model__{k}": v for k, v in config.get("grid_search").items()
            }

            grid_search = GridSearchCV(
                pipeline,
                grid_params,
                cv=3,
                scoring="neg_mean_absolute_percentage_error",
                verbose=0 if not plot else 1,
                n_jobs=-1,
            )
            grid_search.fit(X, y)
            best_pipeline = grid_search.best_estimator_
            results["best_params"] = grid_search.best_params_
            return best_pipeline, results
        else:
            raise KeyError("grid_search config is missing for running cv")

    else:
        X = df[num_columns + cat_columns]
        y = df[target]

        X_transformed = preprocessor.fit_transform(X, y)
        model.fit(X_transformed, y)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Stacking weights: coefficients of the final estimator (Ridge)
    results["estimator_weights"] = dict(
        zip([name for name, _ in estimators], model.final_estimator_.coef_)
    )

    return pipeline, results