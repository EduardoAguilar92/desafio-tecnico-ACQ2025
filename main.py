import argparse
from utils.data_analyzer import DataAnalyzer
from utils.feature_engineer import FeatureEngineer
from utils.model_predictor import ModelPredictor
import re

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Pipeline de Machine Learning')
    parser.add_argument('--train', 
                       type=str, 
                       choices=['True', 'true', 'False', 'false'],
                       default='False',
                       help='Entrenar modelos desde cero (True) o usar modelos existentes (False)')
    
    args = parser.parse_args()
    
    # Convertir string a boolean
    train_flag = args.train.lower() == 'true'
    
    print(f"🚀 Iniciando pipeline con train={train_flag}")
    
    # Crear instancia de DataAnalyzer
    analyzer = DataAnalyzer()

    # Leer el dataset
    dataset_path = './data/raw/new_items_dataset.csv'
    df = analyzer.read_dataset(dataset_path)

    # Resumir los datos
    descriptive_stats, markdown_table_full = analyzer.summarize_data()
    print(markdown_table_full)

    # Identificar y gestionar datos faltantes
    nulls_info, markdown_nulls = analyzer.identify_nulls()
    print(markdown_nulls)

    # Preprocesar datos faltantes
    data_clean, process_nulls_markdown = analyzer.preprocess_nulls(threshold=0.5, verbose=True)
    print(process_nulls_markdown)

    # Procesar outliers
    df_processed, markdown = analyzer.process_outliers(IQR_threshold=1.5)
    print(markdown)

    # Crear instance de FeatureEngineer
    feature_engineer = FeatureEngineer()

    # Crear nuevas características
    new_data = feature_engineer.create_features(df_processed)

    # Detectar correlaciones
    correlation_results = feature_engineer.detect_correlation(new_data)

    # Seleccionar variables
    selected_data = feature_engineer.select_best_features(new_data)
    selection = selected_data['selected_features'] + ["target"]
    filtered_data = new_data[selection]
    filtered_data.to_csv('./data/processed/selected_data.csv', index=False)

    # Crear instancia de ModelPredictor
    model_predictor = ModelPredictor()

    # Dividir los datos
    X_train, X_val, X_test, y_train, y_val, y_test, split_info = model_predictor.split_data(filtered_data)
    print(split_info)

    # Entrenar el modelo con el parámetro desde consola
    train_results = model_predictor.train_model(
        X_train, 
        y_train, 
        train=train_flag,  # Usar el parámetro de consola
        cv_folds=5, 
        cv_scoring='accuracy', 
        save_models=True, 
        models_dir='./models'
    )
    print(train_results)

    # Predecir datos
    predictions = model_predictor.predict(
        X_test,
        model_source='best',
        model_path='./models/training_session_20250701_142622',
        return_probabilities=True,
        return_confidence=True,
        save_results=True,
        output_path='./data/results/predictions.csv'
    )
    print(predictions)

    # Evaluar predicciones
    evaluation_results = model_predictor.evaluate_model(
        y_test, 
        predictions["predictions"], 
        y_probabilities=None, 
        class_names=None, 
        save_plots=False, 
        plots_dir='./evaluation_plots', 
        generate_report=False, 
        report_path=None
    )
    print(evaluation_results)

if __name__ == "__main__":
    main()