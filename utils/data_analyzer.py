import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from utils.statistics import columns_by_dtype, statistical_summary, nulls_barplot, nulls_preprocessing, outlier_processing

class DataAnalyzer:
    """
    Clase para análisis completo de datasets de ventas.
    Incluye lectura, resumen estadístico, gestión de datos faltantes y detección de outliers.
    """
    
    def __init__(self):
        """Inicializa la clase DataAnalyzer."""
        self.data = None
        
    def read_dataset(self, file_path):
        """
        Lee el dataset desde un archivo.
        
        Args:
            file_path (str): Ruta al archivo del dataset
            **kwargs: Argumentos adicionales para pd.read_csv()
            
        Returns:
            pd.DataFrame: Dataset cargado
        """
        print("="*60)
        print("DATASET CARGADO CORRECTAMENTE")
        print("="*60)

        try:
            # Carga el archivo CSV
            self.data = pd.read_csv(file_path)
            self.original_shape = self.data.shape
            
            # Mostrar las primeras filas del dataset
            print("Primeras filas del dataset:")
            print(self.data.head())
        
            return self.data
            
        except Exception as e:
            print(f"❌ Error al cargar el dataset: {str(e)}")
            return None
    
    def summarize_data(self):
        """
        Muestra un resumen estadístico de las variables numéricas del dataset.
        
        Returns:
            pd.DataFrame: Resumen estadístico
        """
        print("="*60)
        print("RESUMEN ESTADÍSTICO")
        print("="*60)
        
        if self.data is None:
            print("❌ No hay datos para resumir. Carga el dataset primero.")
            return None
        
        columns_by_dtype(self.data)
        
        descriptive_stats, markdown_table_full = statistical_summary(self.data, bins=50)

        return descriptive_stats, markdown_table_full

    def identify_nulls(self):
        """
        Identifica y gestiona los datos faltantes en el dataset.
        
        Returns:
            pd.DataFrame: Información sobre los datos faltantes
        """
        print("="*60)
        print("IDENTIFICACIÓN DE DATOS FALTANTES")
        print("="*60)
        if self.data is None:
            print("❌ No hay datos para analizar. Carga el dataset primero.")
            return None
        
        nulls_data, markdown_nulls = nulls_barplot(self.data)

        return nulls_data, markdown_nulls

    def preprocess_nulls(self, threshold=0.5, verbose=True):
        """
        Preprocesa los datos faltantes en el dataset.
        
        Args:
            threshold (float): Umbral para decidir eliminar columna (por defecto 0.5 = 50%).
            
        Returns:
            pd.DataFrame: Dataset preprocesado
        """
        print("="*60)
        print("PREPROCESAMIENTO DE DATOS FALTANTES")
        print("="*60)

        if self.data is None:
            print("❌ No hay datos para procesar. Carga el dataset primero.")
            return None

        data_clean, markdown = nulls_preprocessing(self.data, threshold=threshold, verbose=verbose)

        self.data = data_clean  # Actualizar el DataFrame original
        print(f"Datos preprocesados. Forma original: {self.original_shape}, nueva forma: {data_clean.shape}")

        return data_clean, markdown
    
    def process_outliers(self, IQR_threshold=1.5):
        """
        Detecta y gestiona outliers en el dataset utilizando el método Z-score.
        
        Args:
            z_threshold (float): Umbral Z para considerar un valor como outlier (por defecto 3).
            
        Returns:
            dict: Información sobre los outliers detectados
        """
        print("="*60)
        print("PROCESAMIENTO DE OUTLIERS")
        print("="*60)
        if self.data is None:
            print("❌ No hay datos para procesar. Carga el dataset primero.")
            return None
                
        df_processed, markdown = outlier_processing(self.data, iqr_multiplier=IQR_threshold)
        
        print(f"Datos preprocesados. Forma original: {self.data.shape}, nueva forma: {df_processed.shape}")

        # Actualizar el DataFrame original
        self.data = df_processed

        return df_processed, markdown