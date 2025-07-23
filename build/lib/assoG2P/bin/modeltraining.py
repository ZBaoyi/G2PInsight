
import argparse
import logging
import sys
import warnings
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class ModelTrainer:
    """Main class handling all model training workflows"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def load_data(self, input_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and validate input dataset
        
        Args:
            input_path: Path to input data file (tab-delimited)
            
        Returns:
            Tuple of (features, target)
            
        Raises:
            ValueError: If data format is invalid
        """
        try:
            data = pd.read_csv(input_path, sep='\t')
            if len(data.columns) < 2:
                raise ValueError("Input file must contain at least 2 columns (features and target)")
            
            X = data.iloc[:, 1:-1]  # Features (exclude first column as sample ID)
            y = data.iloc[:, -1]    # Target (last column)
            
            self._validate_data(X, y)
            logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate dataset integrity"""
        if len(X) == 0:
            raise ValueError("Empty feature matrix")
        if len(np.unique(y)) < 2:
            raise ValueError("Target must contain at least 2 classes")
        if len(X) != len(y):
            raise ValueError("Feature/target dimension mismatch")

    def train_model(
        self, 
        model_type: str,
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2
    ) -> Tuple[Any, Optional[np.ndarray]]:
        """Train specified model with hyperparameter tuning
        
        Args:
            model_type: One of ['LightGBM', 'RandomForest', 'XGBoost', 'SVM', 'CatBoost', 'Logistic']
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            
        Returns:
            Tuple of (trained_model, processed_features_for_SHAP)
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
            raise ValueError(f"Unsupported model: {model_type}")
            
        return model_map[model_type](X, y, test_size)
    def _train_lightgbm(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> Tuple[Any, None]:
        """Train LightGBM classifier with grid search"""
        import lightgbm as lgb
        
        # Data splitting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Model configuration
        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=self.random_state,
            verbosity=-1
        )
        
        # Hyperparameter grid
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "num_leaves": [15, 31],
            "max_depth": [3, 5],
            "n_estimators": [100, 200]
        }
        
        # Train with cross-validation
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train
        )
        
        # Evaluation
        self._evaluate_model(best_model, X_test, y_test, "LightGBM")
        return best_model, None

    def _train_randomforest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> Tuple[Any, None]:
        """Train RandomForest classifier"""
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        model = RandomForestClassifier(random_state=self.random_state)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "min_samples_split": [2, 5]
        }
        
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train
        )
        self._evaluate_model(best_model, X_test, y_test, "RandomForest")
        return best_model, None

    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> Tuple[Any, None]:
        """Train XGBoost classifier"""
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        model = xgb.XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        param_grid = {
            "learning_rate": [0.1, 0.2],
            "max_depth": [3, 6],
            "subsample": [0.8, 1.0],
            "n_estimators": [100, 200]
        }
        
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train
        )
        self._evaluate_model(best_model, X_test, y_test, "XGBoost")
        return best_model, None

    def _train_svm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> Tuple[Any, np.ndarray]:
        """Train SVM classifier with feature scaling"""
        from sklearn.svm import SVC
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        model = SVC(
            probability=True,
            random_state=self.random_state
        )
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
        
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train
        )
        self._evaluate_model(best_model, X_test, y_test, "SVM")
        return best_model, X_scaled

    def _train_catboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> Tuple[Any, None]:
        """Train CatBoost classifier"""
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.error("CatBoost not installed! Run: pip install catboost")
            raise
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        model = CatBoostClassifier(
            random_state=self.random_state,
            verbose=False
        )
        param_grid = {
            "iterations": [100, 200],
            "depth": [4, 6],
            "learning_rate": [0.05, 0.1]
        }
        
        best_model = self._perform_grid_search(
            model, param_grid, X_train, y_train
        )
        self._evaluate_model(best_model, X_test, y_test, "CatBoost")
        return best_model, None

    def _train_logistic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float
    ) -> Tuple[Any, np.ndarray]:
        """Train Logistic Regression model"""
        from sklearn.linear_model import LogisticRegression
        
        # Feature scaling
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
            model, param_grid, X_train, y_train
        )
        self._evaluate_model(best_model, X_test, y_test, "Logistic Regression")
        return best_model, X_scaled

    def _perform_grid_search(
        self,
        model: Any,
        param_grid: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """Execute grid search with cross-validation"""
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state
        )
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> None:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"\n{model_name} Evaluation:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    def calculate_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        model_type: str
    ) -> pd.DataFrame:
        """Calculate SHAP-based feature importance"""
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed! Run: pip install shap")
            raise
            
        # Sample data for efficiency
        X_sample = X.sample(
            min(100, X.shape[0]), 
            random_state=self.random_state
        ) if X.shape[0] > 100 else X
        
        # Model-specific explainer
        if model_type in ["LightGBM", "RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        elif model_type == "Logistic":
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:  # SVM
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)
        
        # Process SHAP values
        if isinstance(shap_values, list):
            importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
            
        return pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

def run_training(
    input_path: str,
    model_type: str,
    output_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> int:
    """Main training workflow
    
    Args:
        input_path: Path to input data
        model_type: Model type to train
        output_path: Output CSV path for feature importance
        test_size: Test set proportion (0-1)
        random_state: Random seed
        
    Returns:
        0 on success, 1 on failure
    """
    try:
        logger.info(f"Starting {model_type} training")
        
        trainer = ModelTrainer(random_state)
        X, y = trainer.load_data(input_path)
        
        model, _ = trainer.train_model(model_type, X, y, test_size)
        importance = trainer.calculate_feature_importance(model, X, model_type)
        
        importance.to_csv(output_path, index=False)
        logger.info(f"Feature importance saved to {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1

def cli_main() -> None:
    """Command line interface entry point"""
    parser = argparse.ArgumentParser(
        description="Train machine learning models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Input data file")
    parser.add_argument("-m", "--model", required=True, 
                       choices=["LightGBM", "RandomForest", "XGBoost", "SVM", "CatBoost", "Logistic"],
                       help="Model type")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    sys.exit(run_training(
        input_path=args.input,
        model_type=args.model,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    ))

if __name__ == "__main__":
    cli_main()