import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, log_loss, brier_score_loss
)
import time
import pickle
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import re


class ModelPredictor:
    """
    Clase para la predicción de modelos.
    """
    
    def __init__(self):
        """Inicializa la clase ModelPredictor."""
        self.model = None

    def split_data(self, data, test_size=0.2, val_size=0.2, random_state=42, shuffle_data=True):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba.
        El balanceo reduce todas las clases al nivel de la clase con menor registros.
        
        Args:
            data (pd.DataFrame): Dataset completo a dividir
            test_size (float): Proporción para conjunto de prueba (0.0-1.0)
            val_size (float): Proporción para conjunto de validación (0.0-1.0)
            random_state (int): Semilla para reproducibilidad
            shuffle_data (bool): Mezclar los datos antes de dividir
            
        Returns:
            dict: Diccionario con los conjuntos divididos y información del proceso
        """
        print("="*60)
        print("PARTICION DE DATOS")
        print("="*60)
        
        if data is None or data.empty:
            return None
        
        df = data.copy()
        
        # Identificar la clase minoritaria
        target_counts = df["target"].value_counts()
        min_count = target_counts.min()
        minority_samples = df[df["target"] == target_counts.idxmin()]

        # Obtener muestra aleatoria de la clase mayoritaria
        majority_samples = df[df["target"] == target_counts.idxmax()].sample(
            n=min_count, 
            random_state=random_state
        )

        # Combinar muestras
        sample_data = pd.concat([minority_samples, majority_samples], axis=0)
        sample_data = shuffle(sample_data, random_state=random_state)

        # Separar features y target
        X = sample_data.drop(columns=["target"])
        y = sample_data["target"]

        # Mezclar datos
        if shuffle_data:
            X, y = shuffle(X, y, random_state=random_state)

        # Primera división: separar conjunto de prueba
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
        )
        
        # Segunda división: separar entrenamiento y validación
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
        )

        test_data = pd.concat([X_test, y_test], axis=1)
        test_data.to_csv('./data/processed/test_data.csv', index=False)

        val_data = pd.concat([X_val, y_val], axis=1)
        val_data.to_csv('./data/processed/val_data.csv', index=False)

        train_data = pd.concat([X_train, y_train], axis=1)
        train_data.to_csv('./data/processed/train_data.csv', index=False)

        # Información del proceso
        split_info = {
            'original_shape': df.shape,
            'target_column': "target",
            'total_samples': len(df),
            'balanced_samples': len(sample_data),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_percentage': (len(X_train) / len(X)) * 100,
            'val_percentage': (len(X_val) / len(X)) * 100,
            'test_percentage': (len(X_test) / len(X)) * 100,
            'features_count': X.shape[1],
            'shuffled': shuffle_data,
            'random_state': random_state,
            'test_size': test_size,
            'val_size': val_size,
            'train_size': 1 - test_size - val_size
        }

        # Retornar todos los conjuntos y la información
        return  X_train, X_val, X_test, y_train, y_val, y_test, split_info


    def train_model(self, X_train, y_train, train=True, cv_folds=5, cv_scoring='accuracy', save_models=True, models_dir='./models'):
        """
        Entrena múltiples modelos de machine learning con validación cruzada y guarda los mejores pesos.
        
        Args:
            X_train (pd.DataFrame): Features de entrenamiento
            y_train (pd.Series): Target de entrenamiento
            train (bool): Si entrenar el modelo
            cv_folds (int): Número de folds para validación cruzada
            cv_scoring (str): Métrica para validación cruzada ('accuracy', 'f1', 'precision', 'recall')
            save_models (bool): Si guardar los modelos entrenados
            models_dir (str): Directorio donde guardar los modelos
        
        Returns:
            dict: Diccionario con los resultados de todos los modelos entrenados
        """
        if train:
            print("="*60)
            print("ENTRENAMIENTO DE MÚLTIPLES MODELOS")
            print("="*60)
            
            # Validar entradas
            if X_train is None or y_train is None:
                return {'error': 'Datos de entrenamiento faltantes', 'models': None}
            
            if len(X_train) == 0 or len(y_train) == 0:
                return {'error': 'Datos de entrenamiento vacíos', 'models': None}
            
            # Crear directorio de modelos si no existe
            if save_models:
                os.makedirs(models_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_dir = os.path.join(models_dir, f"training_session_{timestamp}")
                os.makedirs(session_dir, exist_ok=True)
                print(f"💾 Modelos se guardarán en: {session_dir}")
            
            # Determinar tipo de problema
            n_classes = len(np.unique(y_train))
            is_binary = n_classes == 2
            
            print(f"📊 Tipo de problema: {'Binario' if is_binary else 'Multiclase'} ({n_classes} clases)")
            print(f"🔢 Características: {X_train.shape[1]}")
            print(f"📈 Muestras de entrenamiento: {len(X_train):,}")
            print(f"🔄 Validación cruzada: {cv_folds} folds")
            print(f"📏 Métrica: {cv_scoring}")
            print("-" * 60)
            
            # Configurar modelos disponibles
            models_config = {
                'random_forest': {
                    'model': RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'scale_features': False,
                    'name': 'Random Forest'
                },
                'logistic_regression': {
                    'model': LogisticRegression(
                        random_state=42,
                        max_iter=1000
                    ),
                    'scale_features': True,
                    'name': 'Logistic Regression'
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(
                        n_estimators=100,
                        random_state=42
                    ),
                    'scale_features': False,
                    'name': 'Gradient Boosting'
                },
                'svm': {
                    'model': SVC(
                        random_state=42,
                        probability=True
                    ),
                    'scale_features': True,
                    'name': 'Support Vector Machine'
                },
                'decision_tree': {
                    'model': DecisionTreeClassifier(
                        random_state=42
                    ),
                    'scale_features': False,
                    'name': 'Decision Tree'
                }
            }
            
            # Configurar validación cruzada
            if is_binary or n_classes > 2:
                cv_strategy = StratifiedKFold(
                    n_splits=cv_folds, 
                    shuffle=True, 
                    random_state=42
                )
            else:
                cv_strategy = KFold(
                    n_splits=cv_folds, 
                    shuffle=True, 
                    random_state=42
                )
            
            # Diccionario para almacenar resultados
            results = {
                'models': {},
                'summary': {},
                'best_model': None,
                'saved_models_info': {},
                'training_info': {
                    'n_features': X_train.shape[1],
                    'n_train_samples': len(X_train),
                    'n_classes': n_classes,
                    'is_binary': is_binary,
                    'cv_folds': cv_folds,
                    'cv_scoring': cv_scoring,
                    'class_distribution': dict(pd.Series(y_train).value_counts().sort_index()),
                    'feature_names': list(X_train.columns),
                    'timestamp': datetime.now().isoformat(),
                    'session_dir': session_dir if save_models else None
                },
                'error': None
            }
            
            # Entrenar cada modelo
            total_models = len(models_config)
            for i, (model_key, config) in enumerate(models_config.items(), 1):
                try:
                    print(f"🤖 Entrenando modelo {i}/{total_models}: {config['name']}")
                    
                    # Obtener configuración del modelo
                    ml_model = config['model']
                    model_name = config['name']
                    needs_scaling = config['scale_features']
                    
                    # Preparar datos
                    X_train_processed = X_train.copy()
                    scaler = None
                    
                    # Aplicar escalado si es necesario
                    if needs_scaling:
                        print(f"   🔧 Aplicando escalado de características...")
                        scaler = StandardScaler()
                        X_train_processed = pd.DataFrame(
                            scaler.fit_transform(X_train_processed),
                            columns=X_train_processed.columns,
                            index=X_train_processed.index
                        )
                    
                    # Realizar validación cruzada
                    print(f"   📊 Ejecutando validación cruzada...")
                    cv_start_time = time.time()
                    cv_scores = cross_val_score(
                        ml_model, 
                        X_train_processed, 
                        y_train, 
                        cv=cv_strategy,
                        scoring=cv_scoring,
                        n_jobs=-1
                    )
                    cv_time = time.time() - cv_start_time
                    
                    # Entrenar modelo final en todos los datos
                    print(f"   🎯 Entrenando modelo final...")
                    train_start_time = time.time()
                    ml_model.fit(X_train_processed, y_train)
                    training_time = time.time() - train_start_time
                    
                    # Obtener importancia de características si está disponible
                    feature_importance = None
                    if hasattr(ml_model, 'feature_importances_'):
                        feature_importance = dict(zip(
                            X_train.columns, 
                            ml_model.feature_importances_
                        ))
                        feature_importance = dict(
                            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                        )
                    elif hasattr(ml_model, 'coef_'):
                        coef = ml_model.coef_[0] if len(ml_model.coef_.shape) > 1 else ml_model.coef_
                        feature_importance = dict(zip(
                            X_train.columns, 
                            np.abs(coef)
                        ))
                        feature_importance = dict(
                            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                        )
                    
                    # Almacenar resultados del modelo
                    model_results = {
                        'trained_model': ml_model,
                        'scaler': scaler,
                        'cv_results': {
                            'scores': cv_scores.tolist(),
                            'mean_score': cv_scores.mean(),
                            'std_score': cv_scores.std(),
                            'min_score': cv_scores.min(),
                            'max_score': cv_scores.max(),
                            'scoring_metric': cv_scoring,
                            'cv_folds': cv_folds,
                            'cv_time': cv_time
                        },
                        'feature_importance': feature_importance,
                        'model_info': {
                            'model_type': model_key,
                            'model_name': model_name,
                            'training_time': training_time,
                            'feature_scaling': needs_scaling
                        },
                        'error': None
                    }
                    
                    # Guardar modelo si se especifica
                    if save_models:
                        try:
                            # Crear nombres de archivos
                            model_filename = f"{model_key}_model.pkl"
                            scaler_filename = f"{model_key}_scaler.pkl"
                            info_filename = f"{model_key}_info.pkl"
                            
                            model_path = os.path.join(session_dir, model_filename)
                            scaler_path = os.path.join(session_dir, scaler_filename)
                            info_path = os.path.join(session_dir, info_filename)
                            
                            # Guardar modelo
                            with open(model_path, 'wb') as f:
                                pickle.dump(ml_model, f)
                            
                            # Guardar scaler si existe
                            if scaler is not None:
                                with open(scaler_path, 'wb') as f:
                                    pickle.dump(scaler, f)
                            
                            # Guardar información del modelo (sin el modelo en sí)
                            model_info_to_save = {
                                'cv_results': model_results['cv_results'],
                                'feature_importance': feature_importance,
                                'model_info': model_results['model_info'],
                                'feature_names': list(X_train.columns),
                                'target_classes': sorted(y_train.unique().tolist()),
                                'training_timestamp': datetime.now().isoformat()
                            }
                            
                            with open(info_path, 'wb') as f:
                                pickle.dump(model_info_to_save, f)
                            
                            # Almacenar información de archivos guardados
                            saved_info = {
                                'model_path': model_path,
                                'scaler_path': scaler_path if scaler is not None else None,
                                'info_path': info_path,
                                'cv_score': cv_scores.mean(),
                                'saved_at': datetime.now().isoformat()
                            }
                            
                            results['saved_models_info'][model_key] = saved_info
                            print(f"   💾 Modelo guardado: {model_filename}")
                            
                        except Exception as save_error:
                            print(f"   ⚠️ Error guardando modelo: {str(save_error)}")
                            results['saved_models_info'][model_key] = {
                                'error': str(save_error),
                                'saved_at': datetime.now().isoformat()
                            }
                    
                    results['models'][model_key] = model_results
                    
                    # Almacenar en resumen
                    results['summary'][model_key] = {
                        'name': model_name,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'training_time': training_time,
                        'cv_time': cv_time
                    }
                    
                    print(f"   ✅ {model_name}: CV Score = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                    
                except Exception as e:
                    print(f"   ❌ Error en {config['name']}: {str(e)}")
                    results['models'][model_key] = {
                        'trained_model': None,
                        'error': f"Error durante el entrenamiento: {str(e)}"
                    }
                    results['summary'][model_key] = {
                        'name': config['name'],
                        'cv_mean': 0.0,
                        'cv_std': 0.0,
                        'training_time': 0.0,
                        'cv_time': 0.0,
                        'error': str(e)
                    }
                    
                    # Registrar error en saved_models_info
                    if save_models:
                        results['saved_models_info'][model_key] = {
                            'error': f"Training failed: {str(e)}",
                            'saved_at': datetime.now().isoformat()
                        }
            
            # Determinar el mejor modelo
            try:
                valid_models = {k: v for k, v in results['summary'].items() 
                            if 'error' not in v and v['cv_mean'] > 0}
                
                if valid_models:
                    best_model_key = max(valid_models.keys(), 
                                    key=lambda k: valid_models[k]['cv_mean'])
                    results['best_model'] = {
                        'model_key': best_model_key,
                        'model_name': valid_models[best_model_key]['name'],
                        'cv_score': valid_models[best_model_key]['cv_mean'],
                        'cv_std': valid_models[best_model_key]['cv_std']
                    }
                    
                    print("-" * 60)
                    print(f"🏆 MEJOR MODELO: {results['best_model']['model_name']}")
                    print(f"📊 CV Score: {results['best_model']['cv_score']:.4f} (±{results['best_model']['cv_std']:.4f})")
                    
                    # Guardar información del mejor modelo
                    if save_models:
                        best_model_summary = {
                            'best_model_info': results['best_model'],
                            'all_models_summary': results['summary'],
                            'training_info': results['training_info']
                        }
                        
                        summary_path = os.path.join(session_dir, "training_summary.pkl")
                        with open(summary_path, 'wb') as f:
                            pickle.dump(best_model_summary, f)
                        
                        print(f"📋 Resumen guardado: training_summary.pkl")
                        
                else:
                    print("❌ No se pudo entrenar ningún modelo exitosamente")
                    results['error'] = "No se pudo entrenar ningún modelo"
                    
            except Exception as e:
                print(f"❌ Error determinando mejor modelo: {str(e)}")
                results['error'] = f"Error determinando mejor modelo: {str(e)}"
            
            print("="*60)
            
            # Mostrar resumen de archivos guardados
            if save_models and results['saved_models_info']:
                print("\n💾 MODELOS GUARDADOS:")
                for model_key, save_info in results['saved_models_info'].items():
                    if 'error' not in save_info:
                        print(f"   ✅ {model_key}: {os.path.basename(save_info['model_path'])}")
                    else:
                        print(f"   ❌ {model_key}: {save_info['error']}")
                print(f"\n📁 Directorio: {session_dir}")

            markdown_table = """
| Modelo | Mean Score | Std Score | Top 3 Features (Importance) |
|--------|------------|-----------|------------------------------|
"""
    
            # Obtener información de cada modelo
            for model_key, model_data in results['models'].items():
                model_name = model_data['model_info']['model_name']
                mean_score = model_data['cv_results']['mean_score']
                std_score = model_data['cv_results']['std_score']
                
                # Obtener feature importance (si existe)
                feature_importance = model_data.get('feature_importance')
                
                if feature_importance:
                    # Ordenar por importancia y tomar top 3
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    features_text = ", ".join([f"{feat} ({imp:.3f})" for feat, imp in sorted_features])
                else:
                    features_text = "No disponible"
                
                # Agregar fila a la tabla
                markdown_table += f"| {model_name} | {mean_score:.4f} | {std_score:.4f} | {features_text} |\n"
            
            with open('./template/report.md', 'r', encoding='utf-8') as f:
                template = f.read()

            # Reemplazar el marcador de posición en la plantilla
            pattern = r'\{analisis_modelo\}'
            result = re.sub(pattern, markdown_table, template)

            with open('./template/report.md', 'w', encoding='utf-8') as f:
                f.write(result)

            with open('./template/train_results.md', 'w', encoding='utf-8') as f:
                f.write(markdown_table)
                
            return results
        
        else:
            print("="*60)
            print("NO HAY ENTRENAMIENTO")
            print("="*60)

            with open('./template/report.md', 'r', encoding='utf-8') as f:
                template = f.read()

            with open('./template/train_results.md', 'r', encoding='utf-8') as f:
                markdown_table = f.read()

            print(markdown_table)

            # Reemplazar el marcador de posición en la plantilla
            pattern = r'\{analisis_modelo\}'
            result = re.sub(pattern, markdown_table, template)

            with open('./template/report.md', 'w', encoding='utf-8') as f:
                f.write(result)


    def predict(self, X_test, model_source=None, model_path=None, return_probabilities=True, 
           return_confidence=True, save_results=False, output_path=None):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X_test (pd.DataFrame): Features para realizar predicciones
            model_source (str/dict): Fuente del modelo:
                                    - 'best': Usar el mejor modelo de la última sesión
                                    - 'model_key': Usar modelo específico ('random_forest', 'svm', etc.)
                                    - 'path': Ruta a archivo .pkl del modelo
                                    - dict: Resultado directo de train_model()
                                    - None: Usar último modelo entrenado almacenado
            model_path (str): Ruta al directorio que contiene los archivos .pkl del modelo
                            (debe contener: {model_key}_model.pkl, {model_key}_scaler.pkl, {model_key}_info.pkl)
            return_probabilities (bool): Si retornar probabilidades de clase
            return_confidence (bool): Si calcular métricas de confianza
            save_results (bool): Si guardar las predicciones automáticamente
            output_path (str): Ruta donde guardar los resultados (si save_results=True)
        
        Returns:
            dict: Diccionario con predicciones, probabilidades y métricas de confianza
        """
        
        print("="*60)
        print("PREDICCIONES CON MODELO ENTRENADO")
        print("="*60)
        
        # Validar datos de entrada
        if X_test is None or X_test.empty:
            return {'error': 'Datos de prueba vacíos o None', 'predictions': None}
        
        print(f"📊 Datos de prueba: {X_test.shape}")
        
        # Cargar modelo, scaler y metadata
        trained_model = None
        scaler = None
        metadata = {}
        
        try:
            # Caso 1: model_source es un diccionario (resultado directo de entrenamiento)
            if isinstance(model_source, dict) and 'models' in model_source:
                print("📂 Usando modelo desde resultado de entrenamiento...")
                
                if model_source.get('best_model'):
                    best_key = model_source['best_model']['model_key']
                    model_data = model_source['models'][best_key]
                    
                    trained_model = model_data['trained_model']
                    scaler = model_data.get('scaler')
                    metadata = {
                        'model_name': model_data['model_info']['model_name'],
                        'model_type': model_data['model_info']['model_type'],
                        'feature_names': model_source['training_info'].get('feature_names', []),
                        'cv_score': model_data['cv_results']['mean_score']
                    }
                else:
                    return {'error': 'No hay mejor modelo en los resultados', 'predictions': None}
            
            # Caso 2: Usar model_path para cargar archivos específicos
            elif model_path and os.path.exists(model_path):
                print(f"📂 Cargando modelo desde directorio: {model_path}")
                
                # Si model_source es un model_key específico
                if isinstance(model_source, str) and model_source in ['random_forest', 'logistic_regression', 
                                                                    'gradient_boosting', 'svm', 'decision_tree']:
                    model_key = model_source
                else:
                    # Buscar el mejor modelo en el directorio
                    summary_path = os.path.join(model_path, "training_summary.pkl")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'rb') as f:
                            summary = pickle.load(f)
                        model_key = summary['best_model_info']['model_key']
                    else:
                        # Buscar cualquier modelo disponible
                        model_files = [f for f in os.listdir(model_path) if f.endswith('_model.pkl')]
                        if model_files:
                            model_key = model_files[0].replace('_model.pkl', '')
                        else:
                            return {'error': f'No se encontraron modelos en {model_path}', 'predictions': None}
                
                print(f"🤖 Cargando modelo: {model_key}")
                
                # Cargar modelo
                model_file = os.path.join(model_path, f"{model_key}_model.pkl")
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        trained_model = pickle.load(f)
                else:
                    return {'error': f'Archivo de modelo no encontrado: {model_file}', 'predictions': None}
                
                # Cargar scaler si existe
                scaler_file = os.path.join(model_path, f"{model_key}_scaler.pkl")
                if os.path.exists(scaler_file):
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    print("🔧 Scaler cargado")
                
                # Cargar información del modelo
                info_file = os.path.join(model_path, f"{model_key}_info.pkl")
                if os.path.exists(info_file):
                    with open(info_file, 'rb') as f:
                        info = pickle.load(f)
                        metadata = {
                            'model_name': info.get('model_info', {}).get('model_name', f'Modelo {model_key}'),
                            'model_type': model_key,
                            'feature_names': info.get('feature_names', []),
                            'cv_score': info.get('cv_results', {}).get('mean_score', 0),
                            'target_classes': info.get('target_classes', [])
                        }
                    print("📋 Metadata cargada")
            
            # Caso 3: model_source es una ruta completa a archivo
            elif isinstance(model_source, str) and os.path.exists(model_source):
                print(f"📂 Cargando modelo desde archivo: {model_source}")
                
                with open(model_source, 'rb') as f:
                    trained_model = pickle.load(f)
                
                # Buscar scaler y metadata asociados
                base_path = os.path.splitext(model_source)[0]
                scaler_path = base_path.replace('_model', '_scaler') + '.pkl'
                info_path = base_path.replace('_model', '_info') + '.pkl'
                
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                
                if os.path.exists(info_path):
                    with open(info_path, 'rb') as f:
                        info = pickle.load(f)
                        metadata = {
                            'model_name': info.get('model_info', {}).get('model_name', 'Modelo cargado'),
                            'feature_names': info.get('feature_names', []),
                            'cv_score': info.get('cv_results', {}).get('mean_score', 0)
                        }
            
            # Caso 4: Buscar en almacenamiento interno o directorio por defecto
            else:
                # Buscar último modelo entrenado
                if hasattr(self, '_last_training_results') and self._last_training_results:
                    print("📂 Usando último modelo entrenado...")
                    return self.predict(X_test, model_source=self._last_training_results, 
                                    return_probabilities=return_probabilities,
                                    return_confidence=return_confidence,
                                    save_results=save_results, output_path=output_path)
                
                # Buscar en directorio de modelos por defecto
                models_dir = getattr(self, '_models_dir', './models')
                if os.path.exists(models_dir):
                    sessions = [d for d in os.listdir(models_dir) 
                            if d.startswith('training_session_') and os.path.isdir(os.path.join(models_dir, d))]
                    
                    if sessions:
                        latest_session = sorted(sessions)[-1]
                        session_path = os.path.join(models_dir, latest_session)
                        print(f"📂 Usando sesión más reciente: {latest_session}")
                        
                        return self.predict(X_test, model_source=model_source, model_path=session_path,
                                        return_probabilities=return_probabilities,
                                        return_confidence=return_confidence,
                                        save_results=save_results, output_path=output_path)
                
                return {'error': 'No se pudo encontrar ningún modelo para cargar', 'predictions': None}
            
            if trained_model is None:
                return {'error': 'No se pudo cargar el modelo', 'predictions': None}
            
        except Exception as e:
            return {'error': f'Error cargando modelo: {str(e)}', 'predictions': None}
        
        print(f"🤖 Modelo usado: {metadata.get('model_name', 'Desconocido')}")
        print(f"🔧 Escalado requerido: {'Sí' if scaler is not None else 'No'}")
        
        try:
            # Preparar datos de prueba
            X_test_processed = X_test.copy()
            
            # Verificar que las columnas coincidan
            expected_features = metadata.get('feature_names', [])
            if expected_features:
                missing_features = set(expected_features) - set(X_test.columns)
                extra_features = set(X_test.columns) - set(expected_features)
                
                if missing_features:
                    return {
                        'error': f'Faltan características requeridas: {list(missing_features)}',
                        'predictions': None
                    }
                
                if extra_features:
                    print(f"⚠️ Características extra ignoradas: {list(extra_features)}")
                
                # Reordenar columnas para que coincidan con el entrenamiento
                X_test_processed = X_test_processed[expected_features]
            
            # Aplicar escalado si es necesario
            if scaler is not None:
                print("🔧 Aplicando escalado a datos de prueba...")
                X_test_processed = pd.DataFrame(
                    scaler.transform(X_test_processed),
                    columns=X_test_processed.columns,
                    index=X_test_processed.index
                )
            
            # Realizar predicciones
            print("🎯 Generando predicciones...")
            start_time = datetime.now()
            
            predictions = trained_model.predict(X_test_processed)
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            print(f"⏱️ Tiempo de predicción: {prediction_time:.4f} segundos")
            
            # Preparar resultado
            result = {
                'predictions': predictions,
                'prediction_time': prediction_time,
                'model_info': metadata,
                'data_info': {
                    'n_samples': len(X_test),
                    'n_features': X_test_processed.shape[1],
                    'feature_names': list(X_test_processed.columns),
                    'prediction_timestamp': datetime.now().isoformat()
                },
                'preprocessing': {
                    'scaling_applied': scaler is not None,
                    'features_reordered': len(expected_features) > 0,
                    'original_features': list(X_test.columns),
                    'processed_features': list(X_test_processed.columns)
                },
                'error': None
            }
            
            # Obtener probabilidades si se solicita y el modelo las soporta
            if return_probabilities:
                if hasattr(trained_model, 'predict_proba'):
                    print("📊 Calculando probabilidades...")
                    probabilities = trained_model.predict_proba(X_test_processed)
                    class_labels = getattr(trained_model, 'classes_', None)
                    
                    result['probabilities'] = {
                        'raw_probabilities': probabilities,
                        'class_labels': class_labels.tolist() if class_labels is not None else None,
                        'max_probability': np.max(probabilities, axis=1),
                        'predicted_class_probability': probabilities[np.arange(len(probabilities)), 
                                                                    predictions] if len(probabilities.shape) > 1 else probabilities
                    }
                    
                    # Crear DataFrame con probabilidades por clase
                    if class_labels is not None:
                        prob_df = pd.DataFrame(
                            probabilities,
                            columns=[f'prob_class_{label}' for label in class_labels],
                            index=X_test.index
                        )
                        result['probabilities']['probabilities_df'] = prob_df
                    
                else:
                    print("⚠️ El modelo no soporta probabilidades")
                    result['probabilities'] = None
            
            # Calcular métricas de confianza
            if return_confidence and result.get('probabilities'):
                print("🔍 Calculando métricas de confianza...")
                probabilities = result['probabilities']['raw_probabilities']
                
                # Confianza máxima por predicción
                max_probabilities = np.max(probabilities, axis=1)
                
                # Diferencia entre las dos clases más probables
                sorted_probs = np.sort(probabilities, axis=1)
                prob_margins = sorted_probs[:, -1] - sorted_probs[:, -2] if sorted_probs.shape[1] > 1 else sorted_probs[:, -1]
                
                # Entropía de las probabilidades (incertidumbre)
                epsilon = 1e-10  # Para evitar log(0)
                entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
                
                # Normalizar entropía
                max_entropy = np.log(probabilities.shape[1])
                normalized_entropy = entropy / max_entropy
                
                # Categorizar confianza
                confidence_categories = []
                for prob in max_probabilities:
                    if prob >= 0.9:
                        confidence_categories.append('Very High')
                    elif prob >= 0.7:
                        confidence_categories.append('High')
                    elif prob >= 0.5:
                        confidence_categories.append('Medium')
                    else:
                        confidence_categories.append('Low')
                
                confidence_metrics = {
                    'max_probabilities': max_probabilities.tolist(),
                    'probability_margins': prob_margins.tolist(),
                    'entropy': entropy.tolist(),
                    'normalized_entropy': normalized_entropy.tolist(),
                    'confidence_categories': confidence_categories,
                    'summary': {
                        'mean_confidence': float(np.mean(max_probabilities)),
                        'std_confidence': float(np.std(max_probabilities)),
                        'min_confidence': float(np.min(max_probabilities)),
                        'max_confidence': float(np.max(max_probabilities)),
                        'mean_entropy': float(np.mean(normalized_entropy)),
                        'high_confidence_count': sum(1 for cat in confidence_categories if cat in ['High', 'Very High']),
                        'low_confidence_count': sum(1 for cat in confidence_categories if cat == 'Low')
                    }
                }
                
                result['confidence'] = confidence_metrics
            
            # Resumen de predicciones
            unique_predictions, counts = np.unique(predictions, return_counts=True)
            prediction_summary = dict(zip(unique_predictions, counts))
            
            result['prediction_summary'] = {
                'unique_classes': unique_predictions.tolist(),
                'class_counts': prediction_summary,
                'class_percentages': {
                    str(cls): (count / len(predictions)) * 100 
                    for cls, count in prediction_summary.items()
                },
                'most_common_class': unique_predictions[np.argmax(counts)],
                'least_common_class': unique_predictions[np.argmin(counts)]
            }
            
            # Mostrar resumen
            print(f"\n📈 RESUMEN DE PREDICCIONES:")
            for cls, count in prediction_summary.items():
                percentage = (count / len(predictions)) * 100
                print(f"   Clase {cls}: {count:,} predicciones ({percentage:.1f}%)")
            
            if return_probabilities and result['probabilities']:
                avg_confidence = np.mean(result['probabilities']['max_probability'])
                print(f"📊 Confianza promedio: {avg_confidence:.3f}")
            
            # Guardar resultados si se solicita
            if save_results:
                if not output_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"./predictions_{timestamp}.csv"
                
                try:
                    # Crear DataFrame con predicciones
                    df_results = pd.DataFrame({
                        'prediction': predictions
                    }, index=X_test.index)
                    
                    # Añadir probabilidades si están disponibles
                    if result.get('probabilities') and result['probabilities'].get('probabilities_df') is not None:
                        df_results = pd.concat([df_results, result['probabilities']['probabilities_df']], axis=1)
                        df_results['max_probability'] = result['probabilities']['max_probability']
                    
                    # Añadir métricas de confianza
                    if result.get('confidence'):
                        df_results['confidence_category'] = result['confidence']['confidence_categories']
                        df_results['entropy'] = result['confidence']['normalized_entropy']
                    
                    # Guardar archivo
                    if output_path.endswith('.csv'):
                        df_results.to_csv(output_path, index=True)
                    elif output_path.endswith('.pkl'):
                        with open(output_path, 'wb') as f:
                            pickle.dump(result, f)
                    else:
                        # Por defecto, guardar como CSV
                        if not output_path.endswith('.csv'):
                            output_path += '.csv'
                        df_results.to_csv(output_path, index=True)
                    
                    print(f"💾 Predicciones guardadas en: {output_path}")
                    result['saved_file'] = output_path
                    
                except Exception as e:
                    print(f"⚠️ Error guardando predicciones: {str(e)}")
                    result['save_error'] = str(e)
            
            return result
            
        except Exception as e:
            error_msg = f"Error durante predicción: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'predictions': None}
        

    def evaluate_model(self, y_test, y_predicted, y_probabilities=None, class_names=None, 
                  save_plots=False, plots_dir='./evaluation_plots', 
                  generate_report=True, report_path=None):
        """
        Evalúa el rendimiento del modelo con los datos de prueba.
        
        Args:
            y_test (pd.Series/array): Etiquetas reales de prueba
            y_predicted (pd.Series/array): Etiquetas predichas por el modelo
            y_probabilities (array): Probabilidades de predicción (opcional)
            class_names (list): Nombres de las clases (opcional)
            save_plots (bool): Si guardar gráficos de evaluación
            plots_dir (str): Directorio donde guardar los gráficos
            generate_report (bool): Si generar reporte de evaluación
            report_path (str): Ruta donde guardar el reporte
        
        Returns:
            dict: Diccionario con métricas de evaluación completas
        """
        
        print("="*60)
        print("EVALUACIÓN DEL MODELO")
        print("="*60)
        
        # Validaciones iniciales
        if y_test is None or y_predicted is None:
            return {'error': 'Datos de evaluación faltantes'}
        
        # Convertir a arrays si es necesario
        y_test = np.array(y_test)
        y_predicted = np.array(y_predicted)
        
        if len(y_test) == 0 or len(y_predicted) == 0:
            return {'error': 'Datos de evaluación vacíos'}
        
        if len(y_test) != len(y_predicted):
            return {'error': f'Tamaños no coinciden: y_test={len(y_test)}, y_predicted={len(y_predicted)}'}
        
        # Información básica
        n_samples = len(y_test)
        unique_classes = np.unique(np.concatenate([y_test, y_predicted]))
        n_classes = len(unique_classes)
        is_binary = n_classes == 2
        
        print(f"📊 Muestras evaluadas: {n_samples:,}")
        print(f"🎯 Tipo de problema: {'Binario' if is_binary else f'Multiclase ({n_classes} clases)'}")
        print(f"🏷️ Clases: {unique_classes.tolist()}")
        
        # Configurar nombres de clases
        if class_names is None:
            class_names = [f'Clase {cls}' for cls in unique_classes]
        elif len(class_names) != n_classes:
            print(f"⚠️ Número de nombres de clase no coincide, usando nombres automáticos")
            class_names = [f'Clase {cls}' for cls in unique_classes]
        
        # Crear directorio para gráficos si es necesario
        if save_plots:
            import os
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_session_dir = os.path.join(plots_dir, f"evaluation_{timestamp}")
            os.makedirs(plots_session_dir, exist_ok=True)
            print(f"📁 Gráficos se guardarán en: {plots_session_dir}")
        
        # Estructura de resultados
        evaluation_results = {
            'basic_info': {
                'n_samples': n_samples,
                'n_classes': n_classes,
                'is_binary': is_binary,
                'unique_classes': unique_classes.tolist(),
                'class_names': class_names,
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'basic_metrics': {},
            'detailed_metrics': {},
            'confusion_matrix': {},
            'classification_report': {},
            'class_distribution': {},
            'plots_info': {},
            'error': None
        }
        
        try:
            # ===== MÉTRICAS BÁSICAS =====
            print("\n📈 Calculando métricas básicas...")
            
            # Accuracy
            accuracy = accuracy_score(y_test, y_predicted)
            balanced_acc = balanced_accuracy_score(y_test, y_predicted)
            
            # Precision, Recall, F1
            if is_binary:
                precision = precision_score(y_test, y_predicted, average='binary', zero_division=0)
                recall = recall_score(y_test, y_predicted, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_predicted, average='binary', zero_division=0)
            else:
                precision = precision_score(y_test, y_predicted, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_predicted, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_predicted, average='weighted', zero_division=0)
            
            evaluation_results['basic_metrics'] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'matthews_corrcoef': matthews_corrcoef(y_test, y_predicted),
                'cohen_kappa': cohen_kappa_score(y_test, y_predicted)
            }
            
            # ===== MÉTRICAS DETALLADAS =====
            print("📊 Calculando métricas detalladas...")
            
            # Métricas por clase
            precision_per_class = precision_score(y_test, y_predicted, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_predicted, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_predicted, average=None, zero_division=0)
            
            class_metrics = {}
            for i, cls in enumerate(unique_classes):
                class_metrics[f'class_{cls}'] = {
                    'precision': precision_per_class[i] if i < len(precision_per_class) else 0,
                    'recall': recall_per_class[i] if i < len(recall_per_class) else 0,
                    'f1_score': f1_per_class[i] if i < len(f1_per_class) else 0
                }
            
            evaluation_results['detailed_metrics'] = {
                'class_metrics': class_metrics,
                'macro_precision': precision_score(y_test, y_predicted, average='macro', zero_division=0),
                'macro_recall': recall_score(y_test, y_predicted, average='macro', zero_division=0),
                'macro_f1': f1_score(y_test, y_predicted, average='macro', zero_division=0),
                'micro_precision': precision_score(y_test, y_predicted, average='micro', zero_division=0),
                'micro_recall': recall_score(y_test, y_predicted, average='micro', zero_division=0),
                'micro_f1': f1_score(y_test, y_predicted, average='micro', zero_division=0)
            }
            
            # ===== MATRIZ DE CONFUSIÓN =====
            print("🎯 Generando matriz de confusión...")
            
            cm = confusion_matrix(y_test, y_predicted)
            cm_normalized = confusion_matrix(y_test, y_predicted, normalize='true')
            
            evaluation_results['confusion_matrix'] = {
                'raw_matrix': cm.tolist(),
                'normalized_matrix': cm_normalized.tolist(),
                'class_labels': unique_classes.tolist()
            }
            
            # ===== REPORTE DE CLASIFICACIÓN =====
            class_report = classification_report(y_test, y_predicted, 
                                            target_names=class_names, 
                                            output_dict=True, 
                                            zero_division=0)
            evaluation_results['classification_report'] = class_report
            
            # ===== DISTRIBUCIÓN DE CLASES =====
            print("📋 Analizando distribución de clases...")
            
            test_distribution = pd.Series(y_test).value_counts().sort_index()
            pred_distribution = pd.Series(y_predicted).value_counts().sort_index()
            
            evaluation_results['class_distribution'] = {
                'true_distribution': test_distribution.to_dict(),
                'predicted_distribution': pred_distribution.to_dict(),
                'true_percentages': (test_distribution / len(y_test) * 100).to_dict(),
                'predicted_percentages': (pred_distribution / len(y_predicted) * 100).to_dict()
            }
            
            # ===== MÉTRICAS CON PROBABILIDADES =====
            if y_probabilities is not None:
                print("🎲 Calculando métricas con probabilidades...")
                
                prob_metrics = {}
                
                # Log Loss
                try:
                    log_loss_score = log_loss(y_test, y_probabilities)
                    prob_metrics['log_loss'] = log_loss_score
                except Exception:
                    prob_metrics['log_loss'] = None
                
                # Para problemas binarios
                if is_binary and y_probabilities.shape[1] == 2:
                    y_prob_positive = y_probabilities[:, 1]
                    
                    # ROC AUC
                    try:
                        roc_auc = roc_auc_score(y_test, y_prob_positive)
                        prob_metrics['roc_auc'] = roc_auc
                    except Exception:
                        prob_metrics['roc_auc'] = None
                    
                    # Average Precision
                    try:
                        avg_precision = average_precision_score(y_test, y_prob_positive)
                        prob_metrics['average_precision'] = avg_precision
                    except Exception:
                        prob_metrics['average_precision'] = None
                    
                    # Brier Score
                    try:
                        brier_score = brier_score_loss(y_test, y_prob_positive)
                        prob_metrics['brier_score'] = brier_score
                    except Exception:
                        prob_metrics['brier_score'] = None
                
                # Para problemas multiclase
                elif not is_binary:
                    try:
                        roc_auc = roc_auc_score(y_test, y_probabilities, multi_class='ovr', average='weighted')
                        prob_metrics['roc_auc'] = roc_auc
                    except Exception:
                        prob_metrics['roc_auc'] = None
                
                evaluation_results['probability_metrics'] = prob_metrics
            
            # ===== GENERAR GRÁFICOS =====
            if save_plots:
                print("📊 Generando gráficos de evaluación...")
                plots_info = self._generate_evaluation_plots(
                    y_test, y_predicted, y_probabilities, cm, cm_normalized,
                    unique_classes, class_names, is_binary, plots_session_dir
                )
                evaluation_results['plots_info'] = plots_info
            
            # ===== MOSTRAR RESUMEN =====
            print(f"\n🎯 RESULTADOS DE EVALUACIÓN:")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   📊 Balanced Accuracy: {balanced_acc:.4f}")
            print(f"   📊 Precision: {precision:.4f}")
            print(f"   📊 Recall: {recall:.4f}")
            print(f"   📊 F1-Score: {f1:.4f}")
            print(f"   📊 Matthews Correlation: {evaluation_results['basic_metrics']['matthews_corrcoef']:.4f}")
            
            # if y_probabilities is not None and 'probability_metrics' in evaluation_results:
            #     prob_metrics = evaluation_results['probability_metrics']
            #     if prob_metrics.get('roc_auc'):
            #         print(f"   📊 ROC AUC: {prob_metrics['roc_auc']:.4f}")
            #     if prob_metrics.get('log_loss'):
            #         print(f"   📊 Log Loss: {prob_metrics['log_loss']:.4f}")
            
            # # ===== GENERAR REPORTE =====
            # if generate_report:
            #     report_text = self._generate_evaluation_report(evaluation_results)
                
            #     if report_path:
            #         try:
            #             with open(report_path, 'w', encoding='utf-8') as f:
            #                 f.write(report_text)
            #             print(f"📋 Reporte guardado en: {report_path}")
            #             evaluation_results['report_path'] = report_path
            #         except Exception as e:
            #             print(f"⚠️ Error guardando reporte: {str(e)}")
                
            #     evaluation_results['report_text'] = report_text

            report_text = f"""\n🎯 RESULTADOS DE EVALUACIÓN:\n
- 📊 Accuracy: {accuracy:.4f}\n
- 📊 Balanced Accuracy: {balanced_acc:.4f}\n
- 📊 Precision: {precision:.4f}\n
- 📊 Recall: {recall:.4f}\n
- 📊 F1-Score: {f1:.4f}\n
- 📊 Matthews Correlation: {evaluation_results['basic_metrics']['matthews_corrcoef']:.4f}
"""
            
            with open('./template/report.md', 'r', encoding='utf-8') as f:
                template = f.read()

            # Reemplazar el marcador de posición en la plantilla
            pattern = r'\{reporte\}'
            result = re.sub(pattern, report_text, template)

            with open('./template/report.md', 'w', encoding='utf-8') as f:
                f.write(result)
            
            # print("="*60)
            return evaluation_results
            
        except Exception as e:
            error_msg = f"Error durante evaluación: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg}
        
    def _generate_evaluation_plots(self, y_test, y_predicted, y_probabilities, cm, cm_normalized,
                              unique_classes, class_names, is_binary, plots_dir):
        """
        Genera gráficos de evaluación del modelo.
        """
        
        plots_info = {'generated_plots': []}
        
        try:
            # 1. Matriz de Confusión
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Matriz de confusión cruda
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
            axes[0].set_title('Matriz de Confusión (Valores Absolutos)')
            axes[0].set_xlabel('Predicho')
            axes[0].set_ylabel('Real')
            
            # Matriz de confusión normalizada
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
            axes[1].set_title('Matriz de Confusión (Normalizada)')
            axes[1].set_xlabel('Predicho')
            axes[1].set_ylabel('Real')
            
            plt.tight_layout()
            cm_path = f"{plots_dir}/confusion_matrix.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_info['generated_plots'].append(cm_path)
            
            # 2. Distribución de Clases
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            test_counts = pd.Series(y_test).value_counts().sort_index()
            pred_counts = pd.Series(y_predicted).value_counts().sort_index()
            
            # Reindex para asegurar que todas las clases estén presentes
            all_classes = unique_classes
            test_counts = test_counts.reindex(all_classes, fill_value=0)
            pred_counts = pred_counts.reindex(all_classes, fill_value=0)
            
            axes[0].bar(range(len(all_classes)), test_counts.values, alpha=0.7, label='Real')
            axes[0].bar(range(len(all_classes)), pred_counts.values, alpha=0.7, label='Predicho')
            axes[0].set_xticks(range(len(all_classes)))
            axes[0].set_xticklabels([class_names[i] for i in range(len(class_names))])
            axes[0].set_title('Distribución de Clases')
            axes[0].legend()
            axes[0].set_ylabel('Frecuencia')
            
            # Porcentajes
            test_pct = (test_counts / len(y_test) * 100)
            pred_pct = (pred_counts / len(y_predicted) * 100)
            
            axes[1].bar(range(len(all_classes)), test_pct.values, alpha=0.7, label='Real')
            axes[1].bar(range(len(all_classes)), pred_pct.values, alpha=0.7, label='Predicho')
            axes[1].set_xticks(range(len(all_classes)))
            axes[1].set_xticklabels([class_names[i] for i in range(len(class_names))])
            axes[1].set_title('Distribución de Clases (%)')
            axes[1].legend()
            axes[1].set_ylabel('Porcentaje')
            
            plt.tight_layout()
            dist_path = f"{plots_dir}/class_distribution.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots_info['generated_plots'].append(dist_path)
            
            # 3. Gráficos específicos para problemas binarios
            if is_binary and y_probabilities is not None and y_probabilities.shape[1] == 2:
                y_prob_positive = y_probabilities[:, 1]
                
                # ROC Curve
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
                    roc_auc = roc_auc_score(y_test, y_prob_positive)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC Curve (AUC = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Tasa de Falsos Positivos')
                    plt.ylabel('Tasa de Verdaderos Positivos')
                    plt.title('Curva ROC')
                    plt.legend(loc="lower right")
                    plt.grid(alpha=0.3)
                    
                    roc_path = f"{plots_dir}/roc_curve.png"
                    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots_info['generated_plots'].append(roc_path)
                    
                except Exception as e:
                    print(f"⚠️ Error generando curva ROC: {str(e)}")
                
                # Precision-Recall Curve
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_positive)
                    avg_precision = average_precision_score(y_test, y_prob_positive)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                            label=f'PR Curve (AP = {avg_precision:.4f})')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title('Curva Precision-Recall')
                    plt.legend(loc="lower left")
                    plt.grid(alpha=0.3)
                    
                    pr_path = f"{plots_dir}/precision_recall_curve.png"
                    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots_info['generated_plots'].append(pr_path)
                    
                except Exception as e:
                    print(f"⚠️ Error generando curva PR: {str(e)}")
            
            plots_info['plots_directory'] = plots_dir
            return plots_info
            
        except Exception as e:
            return {'error': f'Error generando gráficos: {str(e)}'}

    def _generate_evaluation_report(self, evaluation_results):
        """
        Genera un reporte de texto de la evaluación.
        """
        
        report = []
        report.append("="*60)
        report.append("REPORTE DE EVALUACIÓN DEL MODELO")
        report.append("="*60)
        
        # Información básica
        basic_info = evaluation_results['basic_info']
        report.append(f"\n📊 INFORMACIÓN GENERAL:")
        report.append(f"   • Timestamp: {basic_info['evaluation_timestamp']}")
        report.append(f"   • Muestras evaluadas: {basic_info['n_samples']:,}")
        report.append(f"   • Número de clases: {basic_info['n_classes']}")
        report.append(f"   • Tipo de problema: {'Binario' if basic_info['is_binary'] else 'Multiclase'}")
        report.append(f"   • Clases: {basic_info['unique_classes']}")
        
        # Métricas básicas
        basic_metrics = evaluation_results['basic_metrics']
        report.append(f"\n🎯 MÉTRICAS PRINCIPALES:")
        report.append(f"   • Accuracy: {basic_metrics['accuracy']:.4f}")
        report.append(f"   • Balanced Accuracy: {basic_metrics['balanced_accuracy']:.4f}")
        report.append(f"   • Precision: {basic_metrics['precision']:.4f}")
        report.append(f"   • Recall: {basic_metrics['recall']:.4f}")
        report.append(f"   • F1-Score: {basic_metrics['f1_score']:.4f}")
        report.append(f"   • Matthews Correlation: {basic_metrics['matthews_corrcoef']:.4f}")
        report.append(f"   • Cohen's Kappa: {basic_metrics['cohen_kappa']:.4f}")
        
        # Métricas por clase
        detailed_metrics = evaluation_results['detailed_metrics']
        report.append(f"\n📈 MÉTRICAS POR CLASE:")
        for class_name, metrics in detailed_metrics['class_metrics'].items():
            report.append(f"   • {class_name}:")
            report.append(f"     - Precision: {metrics['precision']:.4f}")
            report.append(f"     - Recall: {metrics['recall']:.4f}")
            report.append(f"     - F1-Score: {metrics['f1_score']:.4f}")
        
        # Métricas de promedio
        report.append(f"\n📊 MÉTRICAS PROMEDIO:")
        report.append(f"   • Macro Precision: {detailed_metrics['macro_precision']:.4f}")
        report.append(f"   • Macro Recall: {detailed_metrics['macro_recall']:.4f}")
        report.append(f"   • Macro F1: {detailed_metrics['macro_f1']:.4f}")
        report.append(f"   • Micro Precision: {detailed_metrics['micro_precision']:.4f}")
        report.append(f"   • Micro Recall: {detailed_metrics['micro_recall']:.4f}")
        report.append(f"   • Micro F1: {detailed_metrics['micro_f1']:.4f}")
        
        # Métricas con probabilidades
        if 'probability_metrics' in evaluation_results:
            prob_metrics = evaluation_results['probability_metrics']
            report.append(f"\n🎲 MÉTRICAS CON PROBABILIDADES:")
            if prob_metrics.get('roc_auc'):
                report.append(f"   • ROC AUC: {prob_metrics['roc_auc']:.4f}")
            if prob_metrics.get('log_loss'):
                report.append(f"   • Log Loss: {prob_metrics['log_loss']:.4f}")
            if prob_metrics.get('average_precision'):
                report.append(f"   • Average Precision: {prob_metrics['average_precision']:.4f}")
            if prob_metrics.get('brier_score'):
                report.append(f"   • Brier Score: {prob_metrics['brier_score']:.4f}")
        
        # Distribución de clases
        class_dist = evaluation_results['class_distribution']
        report.append(f"\n📋 DISTRIBUCIÓN DE CLASES:")
        report.append(f"   Clase   | Real   | Pred   | Real % | Pred %")
        report.append(f"   --------|--------|--------|--------|--------")
        for cls in evaluation_results['basic_info']['unique_classes']:
            real_count = class_dist['true_distribution'].get(cls, 0)
            pred_count = class_dist['predicted_distribution'].get(cls, 0)
            real_pct = class_dist['true_percentages'].get(cls, 0)
            pred_pct = class_dist['predicted_percentages'].get(cls, 0)
            report.append(f"   {str(cls):<7} | {real_count:6,} | {pred_count:6,} | {real_pct:6.1f} | {pred_pct:6.1f}")
        
        # Interpretación
        report.append(f"\n💡 INTERPRETACIÓN:")
        accuracy = basic_metrics['accuracy']
        if accuracy >= 0.9:
            report.append("   • Excelente rendimiento del modelo")
        elif accuracy >= 0.8:
            report.append("   • Buen rendimiento del modelo")
        elif accuracy >= 0.7:
            report.append("   • Rendimiento aceptable del modelo")
        else:
            report.append("   • El modelo necesita mejoras")
        
        balance_diff = abs(basic_metrics['accuracy'] - basic_metrics['balanced_accuracy'])
        if balance_diff > 0.1:
            report.append("   • Posible desbalance en las clases")
        
        if basic_metrics['matthews_corrcoef'] < 0.3:
            report.append("   • Correlación baja entre predicciones y valores reales")
        
        return "\n".join(report)
