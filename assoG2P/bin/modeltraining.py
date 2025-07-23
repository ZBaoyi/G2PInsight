import argparse
import logging
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 配置日志
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ModelTrainer:
    """主类，处理所有模型训练工作流"""
    
    def __init__(self, random_state: int = 42,task_type:str = "classification",test_size:float = 0.2):
        self.random_state = random_state
        self.task_type = task_type
        self.test_size = test_size
    def load_data(self, input_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """加载和验证输入数据集
        
        Args:
            input_path: 输入数据文件路径(tab分隔)
            
        Returns:
            特征和目标的元组
            
        Raises:
            ValueError: 如果数据格式无效
        """
        try:
            data = pd.read_csv(input_path, sep='\t')
            if len(data.columns) < 2:
                raise ValueError("输入文件必须包含至少2列(特征和目标)")
            
            X = data.iloc[:, 1:-1]  # 特征(排除第一列作为样本ID)
            y = data.iloc[:, -1]    # 目标(最后一列)
            
            self._validate_data(X, y)
            logger.info(f"数据加载完成: {X.shape[0]}个样本, {X.shape[1]}个特征")
            return X, y
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """验证数据集完整性"""
        if len(X) == 0:
            raise ValueError("特征矩阵为空")
        if len(X) != len(y):
            raise ValueError("特征和目标维度不匹配")

    def train_model(
        self, 
        model_type: str,
        X: pd.DataFrame, 
        y: pd.Series,
        task_type: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Any, Optional[np.ndarray]]:
        """训练指定模型(带超参数调优)
        
        Args:
            model_type: 模型类型 ['LightGBM', 'RandomForest', 'XGBoost', 'SVM', 'CatBoost', 'Logistic']
            X: 特征矩阵
            y: 目标向量
            task_type: 任务类型 ['classification', 'regression']
            test_size: 测试集比例
            
        Returns:
            训练好的模型和用于SHAP的预处理特征的元组
        """
        model_map = {
            "LightGBM": self._train_lightgbm,
            "RandomForest": self._train_randomforest,
            "XGBoost": self._train_xgboost,
            "SVM": self._train_svm,
            "CatBoost": self._train_catboost,
            "Logistic": self._train_logistic
        }
        
        if model_type not in model_map:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        return model_map[model_type](X, y, task_type, test_size,random_state)

    def _train_lightgbm(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int,
        task_type: str,
        test_size: float
    ) -> Tuple[Any, None]:
        """训练LightGBM模型(分类或回归)"""
        import lightgbm as lgb
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y if task_type == "classification" else None
        )
        
        # 模型配置
        objective = "binary" if task_type == "classification" else "regression"
        metric = "auc" if task_type == "classification" else "rmse"
        
        model = lgb.LGBMClassifier(
            objective=objective,
            random_state=self.random_state,
            verbosity=-1
        ) if task_type == "classification" else lgb.LGBMRegressor(
            objective=objective,
            random_state=self.random_state,
            verbosity=-1
        )
        
        # 超参数网格
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "num_leaves": [15, 31],
            "max_depth": [3, 5],
            "n_estimators": [100, 200]
        }
        
        # 交叉验证训练
        if task_type == "classification" :
            scoring = "roc_auc" 
        else :
            scoring = "neg_mean_squared_error"
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train, random_state,task_type, scoring
        )
        
        # 评估
        self._evaluate_model(best_model, X_test, y_test, "LightGBM", task_type)
        return best_model, None

    def _train_randomforest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int,
        task_type: str,
        test_size: float
    ) -> Tuple[Any, None]:
        """训练随机森林模型(分类或回归)"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if task_type == "classification" else None
        )
        
        model = RandomForestClassifier(random_state=self.random_state) if task_type == "classification" else RandomForestRegressor(random_state=self.random_state)
        
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5]
        }
        
        scoring = "roc_auc" if task_type == "classification" else "neg_mean_squared_error"
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train, random_state,task_type, scoring
        )
        self._evaluate_model(best_model, X_test, y_test, "随机森林", task_type)
        return best_model, None

    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        randon_state: int,
        task_type: str,
        test_size: float
    ) -> Tuple[Any, None]:
        """训练XGBoost模型(分类或回归)"""
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if task_type == "classification" else None
        )
        
        if task_type == "classification":
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                eval_metric="rmse"
            )
            
        param_grid = {
            "learning_rate": [0.1, 0.2],
            "max_depth": [3, 6],
            "subsample": [0.8, 1.0],
            "n_estimators": [100, 200]
        }
        
        scoring = "roc_auc" if task_type == "classification" else "neg_mean_squared_error"
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train, randon_state,task_type, scoring
        )
        self._evaluate_model(best_model, X_test, y_test, "XGBoost", task_type)
        return best_model, None

    def _train_svm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int,
        task_type: str,
        test_size: float
    ) -> Tuple[Any, np.ndarray]:
        """训练SVM模型(分类或回归)"""
        from sklearn.svm import SVC, SVR
        
        # 特征缩放
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if task_type == "classification" else None
        )
        
        if task_type == "classification":
            model = SVC(
                probability=True,
                random_state=self.random_state
            )
        else:
            model = SVR()
            
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
        
        scoring = "roc_auc" if task_type == "classification" else "neg_mean_squared_error"
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train, randon_state,task_type, scoring
        )
        self._evaluate_model(best_model, X_test, y_test, "支持向量机", task_type)
        return best_model, X_scaled

    def _train_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int,
        task_type: str,
        test_size: float
    ) -> Tuple[Any, None]:
        """训练CatBoost模型(分类或回归)"""
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            logger.error("未安装CatBoost! 请运行: pip install catboost")
            raise
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if task_type == "classification" else None
        )
        
        if task_type == "classification":
            model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False
            )
        else:
            model = CatBoostRegressor(
                random_state=self.random_state,
                verbose=False
            )
            
        param_grid = {
            "iterations": [100, 200],
            "depth": [4, 6],
            "learning_rate": [0.05, 0.1]
        }
        
        scoring = "roc_auc" if task_type == "classification" else "neg_mean_squared_error"
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train, randon_state,task_type, scoring
        )
        self._evaluate_model(best_model, X_test, y_test, "CatBoost", task_type)
        return best_model, None

    def _train_logistic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int,
        task_type: str,
        test_size: float
    ) -> Tuple[Any, np.ndarray]:
        """训练逻辑回归模型(仅分类)"""
        if task_type == "regression":
            raise ValueError("逻辑回归不支持回归任务")
            
        from sklearn.linear_model import LogisticRegression
        
        # 特征缩放
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        param_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }
        
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train, randon_state,task_type, "roc_auc"
        )
        self._evaluate_model(best_model, X_test, y_test, "逻辑回归", task_type)
        return best_model, X_scaled

    def _perform_grid_search(
        self,
        model: Any,
        param_grid: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
        task_type: str,
        scoring: str
    ) -> Any:
        """执行带交叉验证的网格搜索"""
        if task_type == "classification":
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=self.random_state
            )
        else:
            cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=self.random_state
            )
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"最佳参数: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        task_type: str
    ) -> None:
        """评估模型性能"""
        if task_type == "classification":
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) == 2 else float('nan'),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }
            
            logger.info(f"\n{model_name} 分类模型评估结果:")
            logger.info(f"准确率: {metrics['accuracy']:.4f}")
            if len(np.unique(y_test)) == 2:
                logger.info(f"AUC值: {metrics['auc']:.4f}")
            logger.info(f"混淆矩阵:\n{metrics['confusion_matrix']}")
        else:
            y_pred = model.predict(X_test)
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
            
            logger.info(f"\n{model_name} 回归模型评估结果:")
            logger.info(f"均方误差(MSE): {metrics['mse']:.4f}")
            logger.info(f"R平方值(R2): {metrics['r2']:.4f}")

    def calculate_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        model_type: str
    ) -> pd.DataFrame:
        """计算基于SHAP的特征重要性"""
        try:
            import shap
        except ImportError:
            logger.error("未安装SHAP! 请运行: pip install shap")
            raise
            
        # 抽样提高效率
        X_sample = X.sample(
            min(100, X.shape[0]), 
            random_state=self.random_state
        ) if X.shape[0] > 100 else X
        
        # 模型特定的解释器
        if model_type in ["LightGBM", "RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_type == "Logistic":
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:  # SVM
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        # 处理SHAP值
        if isinstance(shap_values, list):
            importance = np.abs(shap_values[1]).mean(axis=0) if len(shap_values) == 2 else np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
            
        return pd.DataFrame({
            "feature": X.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

def run_training(
    input_path: str,
    model_type: str,
    output_path: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> int:
    """主训练流程
    
    Args:
        input_path: 输入数据路径
        model_type: 要训练的模型类型
        output_path: 特征重要性输出CSV路径
        task_type: 任务类型 ['classification', 'regression']
        test_size: 测试集比例(0-1)
        random_state: 随机种子
        
    Returns:
        成功返回0，失败返回1
    """
    try:
        logger.info(f"开始 {model_type} 模型训练 ({'分类' if task_type == 'classification' else '回归'}任务)")
        
        trainer = ModelTrainer(random_state)
        X, y = trainer.load_data(input_path)
        
        model, _ = trainer.train_model(model_type, X, y, task_type, test_size)
        importance = trainer.calculate_feature_importance(model, X, model_type)
        
        importance.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"特征重要性已保存至 {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        return 1

def cli_main() -> None:
    """命令行接口入口点"""
    parser = argparse.ArgumentParser(
        description="训练机器学习模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="输入数据文件路径")
    parser.add_argument("-m", "--model", required=True, 
                       choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],
                       help="模型类型")
    parser.add_argument( "--task_type", required=True,
                       choices=["classification", "regression"],
                       help="任务类型(分类或回归)")
    parser.add_argument("-o", "--output", required=True, help="输出CSV文件路径")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 检查逻辑回归是否用于回归任务
    if args.model == "Logistic" and args.task_type == "regression":
        logger.error("错误: 逻辑回归不支持回归任务")
        sys.exit(1)
    
    sys.exit(run_training(
        input_path=args.input,
        model_type=args.model,
        output_path=args.output,
        task_type=args.task_type,
        test_size=args.test_size,
        random_state=args.random_state
    ))

if __name__ == "__main__":
    # 配置中文日志输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    cli_main()