import pandas as pd
from utils.statistics import visualize_correlations
import re
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


class FeatureEngineer:
    """
    Clase para ingeniería de características en el datasets.
    Incluye creación de nuevas características, transformación y selección de variables.
    """
    
    def __init__(self):
        """Inicializa la clase FeatureEngineer."""
        self.data = None

    def create_features(self, data):
        """
        Crea nuevas características a partir de las existentes en el dataset.
        """
        print("="*60)
        print("CREACIÓN DE NUEVAS CARACTERÍSTICAS")
        print("="*60)

        if data is None:
            print("❌ No hay datos para crear características. Carga el dataset primero.")
            return None

        new_data = data.copy()

        new_data["shipping_admits_pickup"] = (new_data["shipping_admits_pickup"] == "True")*1
        new_data["shipping_is_free"] = (new_data["shipping_is_free"] == "True")*1

        # Crear variables de tiempo
        new_data["new_date"] = pd.to_datetime(new_data["date_created"], utc=True)
        new_data['day_of_week'] = new_data['new_date'].dt.dayofweek
        new_data['month'] = new_data['new_date'].dt.month
        new_data['year'] = new_data['new_date'].dt.year
        new_data['hour'] = new_data['new_date'].dt.hour
        new_data["images"] = new_data["pictures"].apply(lambda x: len(eval(x)))
        new_data["free_seller"] = (new_data["seller_loyalty"] == "free")*1
        new_data["bronze_seller"] = (new_data["seller_loyalty"] == "bronze")*1
        new_data["silver_seller"] = (new_data["seller_loyalty"] == "silver")*1
        new_data["gold_seller"] = (new_data["seller_loyalty"] == "gold")*1
        new_data["gold_premium_seller"] = (new_data["seller_loyalty"] == "gold_premium")*1
        new_data["gold_pro_seller"] = (new_data["seller_loyalty"] == "gold_pro")*1
        new_data["gold_special_seller"] = (new_data["seller_loyalty"] == "gold_special")*1

        new_data["target"] = (new_data["sold_quantity"] > 0)*1

        new_data.drop(["sold_quantity", "base_price", "date_created"], axis=1, inplace=True)

        self.data = new_data

        markdown = f"""- **new_date:** Transformación de la columna `date_created` a tipo datetime.
- **day_of_week:** Extracción del día de la semana de la columna `new_date`.
- **month:** Extracción del mes de la columna `new_date`.
- **year:** Extracción del año de la columna `new_date`.
- **hour:** Extracción de la hora de la columna `new_date`.
- **images:** Cantidad de imágenes en la columna `pictures`.
- **target:** Variable objetivo para predecir si tendrá venta o no la publicación

        """

        with open('./template/report.md', 'r', encoding='utf-8') as f:
            template = f.read()

        # Reemplazar el marcador de posición en la plantilla
        pattern = r'\{proceso_variables\}'
        result = re.sub(pattern, markdown, template)

        with open('./template/report.md', 'w', encoding='utf-8') as f:
            f.write(result)

        return new_data
    
    def detect_correlation(self, data):
        """
        Detecta las características relevantes en el dataset.
        """

        print("="*60)
        print("DETECCIÓN DE CORRELACION")
        print("="*60)

        results = visualize_correlations(data=data)
        return results
    
    def select_best_features(self, data):
        """
        Selecciona las mejores características usando Random Forest Feature Importance.
        
        Args:
            data (pd.DataFrame): Dataset completo con target incluido
            
        Returns:
            dict: Diccionario con las mejores características y gráfico de importancia
        """
        
        print("="*60)
        print("SELECCIÓN DE CARACTERÍSTICAS - FEATURE IMPORTANCE")
        print("="*60)
        
        # Validaciones iniciales
        if data is None or data.empty:
            return {'error': 'Dataset vacío o None', 'selected_features': []}
                
        # Detectar columna target automáticamente
        possible_targets = ['target', 'y', 'label', 'class', 'outcome', 'prediction']
        target_column = None
        
        for col in possible_targets:
            if col in data.columns:
                target_column = col
                break
        
        if target_column is None:
            return {'error': 'No se pudo detectar la columna target automáticamente', 'selected_features': []}
        
        print(f"🎯 Target detectado: '{target_column}'")
        
        # Separar features y target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Solo características numéricas para simplificar
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            return {'error': 'No se encontraron características numéricas', 'selected_features': []}
        
        X_numeric = X[numeric_features]
        
        print(f"📊 Características numéricas encontradas: {len(numeric_features)}")
        print(f"📈 Muestras: {len(X_numeric):,}")
        print(f"🎯 Clases únicas: {len(y.unique())}")
        
        try:
            # Entrenar Random Forest para obtener importancias
            print("\n🌳 Entrenando Random Forest para Feature Importance...")
            
            rf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            rf.fit(X_numeric, y)
            
            # Obtener importancias
            feature_importance = rf.feature_importances_
            feature_names = X_numeric.columns
            
            # Crear DataFrame con importancias
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"✅ Feature Importance calculado exitosamente")
            
            # Calcular umbral dinámico para selección
            mean_importance = np.mean(feature_importance)
            std_importance = np.std(feature_importance)
            threshold = mean_importance
            
            # Seleccionar características importantes
            selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
            
            # Asegurar que tengamos al menos 3 características
            if len(selected_features) < 3:
                selected_features = importance_df.head(3)['feature'].tolist()
            
            # Limitar a máximo 2/3 de las características
            max_features = max(3, (len(numeric_features) * 2) // 3)
            if len(selected_features) > max_features:
                selected_features = importance_df.head(max_features)['feature'].tolist()
            
            print(f"🎯 Características seleccionadas: {len(selected_features)} de {len(numeric_features)}")
            print(f"📏 Umbral de importancia: {threshold:.4f}")
            
            # Crear gráfico de Feature Importance
            plt.figure(figsize=(12, max(6, len(numeric_features) * 0.4)))
            
            # Configurar colores (seleccionadas vs no seleccionadas)
            colors = ['#2E8B57' if feat in selected_features else '#D3D3D3' 
                    for feat in importance_df['feature']]
            
            # Crear gráfico de barras horizontal
            bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
            
            # Personalizar gráfico
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importancia de la Característica')
            plt.title('Feature Importance - Random Forest\n(Verde: Seleccionadas, Gris: No seleccionadas)', 
                    fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Agregar línea de umbral
            plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                    label=f'Umbral: {threshold:.4f}')
            plt.legend()
            
            # Agregar valores en las barras
            for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
                plt.text(importance + max(importance_df['importance']) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{importance:.4f}', 
                        va='center', fontsize=9)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar gráfico
            plt.savefig('./docs/feature_importance_plot.png', dpi=300, bbox_inches='tight')
            
            # plt.show()
            
            # Preparar resultados
            selection_results = {
                'selected_features': selected_features,
                'target_column': target_column,
                'total_features': len(numeric_features),
                'n_selected': len(selected_features),
                'selection_percentage': (len(selected_features) / len(numeric_features)) * 100,
                'importance_threshold': threshold,
                'feature_importance': dict(zip(feature_names, feature_importance)),
                'importance_df': importance_df,
                'method_used': 'Random Forest Feature Importance',
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'statistics': {
                    'mean_importance': mean_importance,
                    'std_importance': std_importance,
                    'max_importance': np.max(feature_importance),
                    'min_importance': np.min(feature_importance),
                    'top_feature': importance_df.iloc[0]['feature'],
                    'top_importance': importance_df.iloc[0]['importance']
                },
                'error': None
            }
            
            markdown = f"""- Importancia promedio: {mean_importance:.4f}
- Desviación estándar: {std_importance:.4f}
- Umbral de selección: {threshold:.4f}
- Porcentaje seleccionado: {selection_results['selection_percentage']:.1f}%

    | Característica | Importancia | Estado |
    |----------------|-------------|--------|
"""
            
            # Mostrar top características
            print(f"\n🏆 TOP CARACTERÍSTICAS SELECCIONADAS:")
            for i, (_, row) in enumerate(importance_df.head(len(selected_features)).iterrows(), 1):
                feature = row['feature']
                importance = row['importance']
                status = "✅ SELECCIONADA" if feature in selected_features else "❌ No seleccionada"
                print(f"   {i:2d}. {feature:<25} | {importance:.4f} | {status}")
                markdown += f"  |{feature:<25} | {importance:.4f} | {status} |\n"
            
            # Mostrar estadísticas
            print(f"\n📊 ESTADÍSTICAS DE IMPORTANCIA:")
            print(f"   • Característica más importante: {selection_results['statistics']['top_feature']} ({selection_results['statistics']['top_importance']:.4f})")
            print(f"   • Importancia promedio: {mean_importance:.4f}")
            print(f"   • Desviación estándar: {std_importance:.4f}")
            print(f"   • Umbral de selección: {threshold:.4f}")
            print(f"   • Porcentaje seleccionado: {selection_results['selection_percentage']:.1f}%")
            
            # Información adicional sobre las características seleccionadas
            selected_importance_sum = sum([selection_results['feature_importance'][feat] for feat in selected_features])
            total_importance_sum = sum(feature_importance)
            importance_coverage = (selected_importance_sum / total_importance_sum) * 100
            
            selection_results['importance_coverage'] = importance_coverage
            
            print(f"   • Cobertura de importancia: {importance_coverage:.1f}%")
            print(f"     (Las características seleccionadas representan el {importance_coverage:.1f}% de la importancia total)")
            
            print("="*60)

            markdown += "\n\n   ![Gráfico de Importancia](../docs/feature_importance_plot.png)"

            with open('./template/report.md', 'r', encoding='utf-8') as f:
                template = f.read()

            # Reemplazar el marcador de posición en la plantilla
            pattern = r'\{proceso_seleccion_variables\}'
            result = re.sub(pattern, markdown, template)

            with open('./template/report.md', 'w', encoding='utf-8') as f:
                f.write(result)
                return selection_results
            
        except Exception as e:
            error_msg = f"Error durante selección de características: {str(e)}"
            print(f"❌ {error_msg}")
            return {'error': error_msg, 'selected_features': []}