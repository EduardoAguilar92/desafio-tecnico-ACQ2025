import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import re
from scipy.stats import chi2_contingency, pearsonr, spearmanr


def columns_by_dtype(data):
    """
    Función para contar el número de columnas por tipo de dato en un DataFrame.
    
    Args:
    data: El DataFrame del cual se quieren contar los tipos de datos.
    
    Return:
    Series: Conteo de columnas por tipo de dato.
    """

    # Leer la plantilla del reporte
    with open('./template/report_template.md', 'r', encoding='utf-8') as f:
        template = f.read()

    # Preparar los valores de reemplazo
    total_registros = str(data.shape[0])
    total_columnas = str(data.shape[1])
    tamanio_archivo = f"{data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"

    # Reemplazar los marcadores de posición (CORREGIDO)
    result = template  # Empezar con el template original
    
    # Reemplazar cada placeholder
    result = re.sub(r'\{total_registros\}', total_registros, result)
    result = re.sub(r'\{total_columnas\}', total_columnas, result)
    result = re.sub(r'\{tamanio_archivo\}', tamanio_archivo, result)
    
    # Guardar el resultado
    with open('./template/report.md', 'w', encoding='utf-8') as f:
        f.write(result)

    # Clasificar columnas por tipo
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    boolean_cols = data.select_dtypes(include=['bool']).columns.tolist()

    # Crear un diccionario con los conteos
    dtype_counts = {
        'Numérico': len(numeric_cols),
        'Categórico': len(categorical_cols),
        'Fecha/Hora': len(datetime_cols),
        'Booleano': len(boolean_cols)
    }
    
    # Añadir el gráfico a la tabla
    columnas_por_tipo = "\n\n" + "![Gráfico de Tipos de Datos](../docs/grafico_tipos_de_datos.png)\n\n"

    with open('./template/report.md', 'r', encoding='utf-8') as f:
        template = f.read()

    # Reemplazar el marcador de posición en la plantilla
    pattern = r'\{columnas_por_tipo\}'
    result = re.sub(pattern, columnas_por_tipo, template)

    with open('./template/report.md', 'w', encoding='utf-8') as f:
        f.write(result)

    # Convertir el diccionario a una Serie
    dtype_counts = pd.Series(dtype_counts)

    # Estilo visual
    sns.set_style("whitegrid")

    # Crear figura
    plt.figure(figsize=(12, 7))

    # Crear gráfico de barras
    plot = sns.barplot(
        x=dtype_counts.index.astype(str),
        y=dtype_counts.values,
        palette=sns.color_palette("viridis", len(dtype_counts)),
        edgecolor='black'
    )

    # Añadir etiquetas encima de las barras
    for patch in plot.patches:
        height = patch.get_height()
        plot.text(
            x=patch.get_x() + patch.get_width() / 2,
            y=height + max(dtype_counts.values)*0.01,  # margen proporcional
            s=f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=14,
            color='black',
            weight='bold'
        )

    # Títulos y etiquetas
    plot.set_title('Número de Columnas por Tipo de Dato', fontsize=18, weight='bold')
    plot.set_xlabel('Tipo de Dato', fontsize=16)
    plot.set_ylabel('Cantidad de Columnas', fontsize=16)

    # Estética del eje X
    plot.set_xticklabels(plot.get_xticklabels(), fontsize=16)

    plot.set_yticks(range(0, int(max(dtype_counts.values)) + 2, 2))

    # Estética del eje Y
    plot.set_yticklabels(plot.get_yticklabels(), fontsize=14)

    # Quita borde superior y derecho
    sns.despine()

    # Ajuste del diseño
    plt.tight_layout()

    # Guardar y mostrar
    plt.savefig('./docs/grafico_tipos_de_datos.png', dpi=300, bbox_inches='tight')

    # plt.show()

    return dtype_counts

def statistical_summary(data, bins=50):
    """
    Genera un análisis descriptivo y visual para las columnas numéricas.

    Para cada columna numérica, calcula estadísticas y genera un histograma y
    un boxplot.

    Args:
        data (pd.DataFrame): DataFrame de entrada.
        bins (int): Número de bins para los histogramas.

    Returns:
        - descriptive_stats (dict): Diccionario con estadísticas descriptivas.
        - markdown_table_full (str): Tabla en formato Markdown con las estadísticas.
    """
    # Seleccionar solo columnas numéricas
    num_cols = data.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        print("No se encontraron columnas numéricas.")
        return {}, ""

    # Calcular estadísticas descriptivas para datos completos
    descriptive_stats = {}
    for col in num_cols:
        base_stats = data[col].describe()
        # Agregar percentiles adicionales (1%, 5%, 10%, 90%, 95%, 99%)
        p1 = data[col].quantile(0.01)
        p5 = data[col].quantile(0.05)
        p10 = data[col].quantile(0.10)
        p90 = data[col].quantile(0.90)
        p95 = data[col].quantile(0.95)
        p99 = data[col].quantile(0.99)
        
        # Crear diccionario extendido con los nuevos percentiles
        extended_stats = base_stats.to_dict()
        extended_stats['1%'] = p1
        extended_stats['5%'] = p5
        extended_stats['10%'] = p10
        extended_stats['90%'] = p90
        extended_stats['95%'] = p95
        extended_stats['99%'] = p99
        
        # Convertir de vuelta a Series para mantener compatibilidad
        descriptive_stats[col] = pd.Series(extended_stats)

    # Crear la figura y los subplots
    n_rows = len(num_cols)
    n_cols = 2
    figsize = (14, 5 * n_rows)
    gridspec_kw = {'width_ratios': [4, 1]}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             gridspec_kw=gridspec_kw, squeeze=False)

    fig.suptitle('Análisis Descriptivo de Columnas Numéricas', fontsize=16, weight='bold', y=1.02)

    # Iterar sobre cada columna numérica y graficar
    for i, col in enumerate(num_cols):
        # Histograma con la distribución completa
        ax_hist_full = axes[i, 0]
        sns.histplot(data=data, x=col, bins=bins, ax=ax_hist_full, kde=True)
        ax_hist_full.set_xlabel("Valor")
        ax_hist_full.set_ylabel("Frecuencia")
        ax_hist_full.set_title(f"Distribución de '{col}'")

        # Boxplot con la distribución completa
        ax_box_full = axes[i, 1]
        sns.boxplot(y=data[col], ax=ax_box_full)
        ax_box_full.set_title(f"Boxplot de '{col}'")
        ax_box_full.set_ylabel("")
        ax_box_full.set_xticks([])

    # Quita borde superior y derecho
    sns.despine()

    # Ajuste del diseño
    plt.tight_layout()
    
    # Descomenta la siguiente línea para guardar la imagen
    plt.savefig('./docs/distribucion_numericas.png', dpi=300, bbox_inches='tight')
    
    # plt.show()

    # Función auxiliar para crear tabla Markdown
    def create_markdown_table(stats_dict):
        """Crea una tabla Markdown a partir de un diccionario de estadísticas."""
        stats_df = pd.DataFrame(stats_dict).round(2)
        
        # Traducir las etiquetas de estadísticas al español
        translation_map = {
            'count': 'Registros',
            'mean': 'Media',
            'std': 'Desviación Estándar',
            'min': 'Mínimo',
            '50%': 'Mediana',
            'max': 'Máximo'
        }
        
        # Aplicar traducción
        stats_df.index = stats_df.index.map(lambda x: translation_map.get(x, x))
        
        # Reordenar las filas para que los percentiles aparezcan en orden lógico
        desired_order = ['Registros', 'Media', 'Mediana', 'Desviación Estándar', 'Mínimo', '1%', '5%', '10%', '25%', '75%', '90%', '95%', '99%', 'Máximo']
        # Filtrar solo las estadísticas que existen en el DataFrame
        existing_order = [stat for stat in desired_order if stat in stats_df.index]
        # Agregar cualquier estadística adicional que no esté en el orden deseado
        remaining_stats = [stat for stat in stats_df.index if stat not in existing_order]
        final_order = existing_order + remaining_stats
        
        # Reordenar el DataFrame
        stats_df = stats_df.reindex(final_order)
        
        # Creación manual de la tabla Markdown
        header_list = ["Estadística"] + list(stats_df.columns)
        header_str = "| " + " | ".join(header_list) + " |"
        separator_str = "|:---| " + " | ".join([":---"] * len(stats_df.columns)) + " |"
        body_lines = [f"| **{index}** | " + " | ".join(map(str, row.values)) + " |" 
                     for index, row in stats_df.iterrows()]
        markdown_table = f"{header_str}\n{separator_str}\n" + "\n".join(body_lines)

        # Agregar el gráfico de distribución al reporte
        markdown_table += "\n\n![Distribución de Columnas Numéricas](../docs/distribucion_numericas.png)\n\n"

        with open('./template/report.md', 'r', encoding='utf-8') as f:
            template = f.read()

        # Reemplazar el marcador de posición en la plantilla
        pattern = r'\{resumen_numerico\}'
        result = re.sub(pattern, markdown_table, template)

        with open('./template/report.md', 'w', encoding='utf-8') as f:
            f.write(result)
        
        return markdown_table

    # Crear tabla Markdown
    markdown_table_full = create_markdown_table(descriptive_stats)
    
    return descriptive_stats, markdown_table_full

def nulls_barplot(data):
    """
    Crea gráficos de barras horizontales para visualizar la cantidad y porcentaje de nulos por columna.
    Genera automáticamente ambos tipos de visualización.
    
    Args:
    - data (DataFrame): El DataFrame del cual se quieren contar los valores nulos.
    
    Returns:
    - null_df (DataFrame): DataFrame con conteo y porcentaje de nulos por columna.
    - markdown_nulos (str): Resumen en formato Markdown de los nulos encontrados.
    """

    # Número de valores nulos por columna
    null_counts = data.isnull().sum().sort_values(ascending=False)

    # Porcentaje de nulos
    null_percentage = (null_counts / len(data)) * 100

    # Crear un DataFrame para los nulos
    null_df = pd.DataFrame({
        'Conteo': null_counts,
        'Porcentaje': null_percentage
    })

    # Filtrar solo columnas que tienen al menos un valor nulo
    null_df_filtered = null_df[null_df['Conteo'] > 0]
    
    if null_df_filtered.empty:
        print("✅ No se encontraron valores nulos en el dataset")
        return null_df

    # Configurar el estilo
    sns.set_style("whitegrid")
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # === GRÁFICO 1: CONTEO DE NULOS ===
    sns.barplot(
        x=null_df_filtered['Conteo'],
        y=null_df_filtered.index,
        orient='h',
        ax=ax1,
        palette='Reds_r'
    )

    # Añadir etiquetas de valores en el gráfico de conteo
    for patch in ax1.patches:
        ax1.text(
            patch.get_width() + (null_df_filtered['Conteo'].max() * 0.01),
            patch.get_y() + patch.get_height() / 2.,
            f'{int(patch.get_width()):,}',  # Formato con comas para miles
            ha='left',
            va='center',
            fontsize=11,
            color='black',
            weight='bold'
        )

    # Configurar el primer gráfico
    ax1.set_title('Cantidad de Valores Nulos por Columna', fontsize=16, weight='bold', pad=20)
    ax1.set_xlabel('Cantidad de Nulos', fontsize=12)
    ax1.set_ylabel('Nombre de la Columna', fontsize=12)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xlim(0, null_df_filtered['Conteo'].max() * 1.15)

    # === GRÁFICO 2: PORCENTAJE DE NULOS ===
    sns.barplot(
        x=null_df_filtered['Porcentaje'],
        y=null_df_filtered.index,
        orient='h',
        ax=ax2,
        palette='Oranges_r'
    )

    # Añadir etiquetas de valores en el gráfico de porcentaje
    for patch in ax2.patches:
        ax2.text(
            patch.get_width() + (null_df_filtered['Porcentaje'].max() * 0.01),
            patch.get_y() + patch.get_height() / 2.,
            f'{patch.get_width():.1f}%',
            ha='left',
            va='center',
            fontsize=11,
            color='black',
            weight='bold'
        )

    # Configurar el segundo gráfico
    ax2.set_title('Porcentaje de Valores Nulos por Columna', fontsize=16, weight='bold', pad=20)
    ax2.set_xlabel('Porcentaje de Nulos (%)', fontsize=12)
    ax2.set_ylabel('', fontsize=12)  # Ocultar ylabel del segundo gráfico
    ax2.tick_params(axis='y', labelsize=11)
    ax2.tick_params(axis='x', labelsize=11)
    ax2.set_xlim(0, null_df_filtered['Porcentaje'].max() * 1.15)

    # Título general de la figura
    fig.suptitle('Análisis de Valores Nulos en el Dataset', fontsize=18, weight='bold', y=0.98)

    # Quitar bordes superiores y derechos
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)

    # Ajustar el diseño
    plt.tight_layout()

    # Guardar la imagen
    plt.savefig('./docs/analisis_nulos_completo.png', dpi=300, bbox_inches='tight')

    # Mostrar el gráfico
    # plt.show()
    
    total_columns = len(data.columns)
    columns_with_nulls = len(null_df_filtered)
    total_nulls = null_df['Conteo'].sum()
    total_cells = len(data) * len(data.columns)
    overall_null_percentage = (total_nulls / total_cells) * 100
    
    # print(f"Total de columnas: {total_columns}")
    # print(f"Columnas con valores nulos: {columns_with_nulls} ({columns_with_nulls/total_columns*100:.1f}%)")
    # print(f"Total de valores nulos: {total_nulls:,}")
    # print(f"Porcentaje general de nulos: {overall_null_percentage:.2f}%")

    markdown_nulos = f"""
- **Columnas con valores nulos**: {columns_with_nulls} ({columns_with_nulls/total_columns*100:.1f}%)
- **Total de valores nulos**: {total_nulls:,}
- **Porcentaje general de nulos**: {overall_null_percentage:.2f}%
"""
    
    if columns_with_nulls > 0:
        # print(f"\nTop 5 columnas con más nulos:")
        markdown_nulos += "\n\n### 🔍 Top 5 Columnas con Más Nulos\n\n"
        for i, (col, row) in enumerate(null_df_filtered.head(5).iterrows(), 1):
            # print(f"  {i}. {col}: {row['Conteo']:,} nulos ({row['Porcentaje']:.1f}%)")
            markdown_nulos += f"{i}. **{col}**: {row['Conteo']:,} nulos ({row['Porcentaje']:.1f}%)\n"

    markdown_nulos += "\n\n![Análisis de Nulos](../docs/analisis_nulos_completo.png)\n\n"

    with open('./template/report.md', 'r', encoding='utf-8') as f:
        template = f.read()

    # Reemplazar el marcador de posición en la plantilla
    pattern = r'\{nulos\}'
    result = re.sub(pattern, markdown_nulos, template)

    with open('./template/report.md', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return null_df, markdown_nulos

def nulls_preprocessing(data, threshold=0.5, verbose=True):
    """
    Preprocesa los valores nulos en el DataFrame.
    
    Para cada columna:
    - Si % de nulos >= threshold: elimina la columna completa
    - Si % de nulos < threshold: elimina solo los registros con nulos en esa columna
    
    Args:
        data (pd.DataFrame): DataFrame con valores nulos.
        threshold (float): Umbral para decidir eliminar columna (por defecto 0.5 = 50%).
        verbose (bool): Mostrar información del proceso.
        
    Returns:
        pd.DataFrame: DataFrame con valores nulos preprocesados.
        markdown: Resumen del proceso en formato Markdown.
    """
    
    if data is None or data.empty:
        print("❌ DataFrame vacío o None")
        return data
    
    # Crear copia para no modificar el original
    df_processed = data.copy()
    original_shape = df_processed.shape
    
    # Listas para tracking
    columns_dropped = []
    columns_cleaned = []        
    
    # Procesar cada columna individualmente
    for col in data.columns:
        null_count = df_processed[col].isnull().sum()
        total_rows = len(df_processed)
        null_percentage = (null_count / total_rows) if total_rows > 0 else 0
        
        if null_count == 0:
            # No hay nulos, no hacer nada
            if verbose:
                print(f"✅ {col:<25}: Sin nulos")
            continue
            
        elif null_percentage >= threshold:
            # Muchos nulos: eliminar columna completa
            df_processed = df_processed.drop(columns=[col])
            columns_dropped.append({
                'columna': col,
                'nulos': null_count,
                'porcentaje': null_percentage * 100,
                'total_filas_momento': total_rows
            })
            if verbose:
                print(f"🗑️  {col:<25}: Columna eliminada ({null_count:,} nulos, {null_percentage*100:.1f}%)")
        else:
            # Pocos nulos: eliminar solo las filas con nulos en esta columna
            rows_before = len(df_processed)
            df_processed = df_processed.dropna(subset=[col])
            rows_after = len(df_processed)
            rows_dropped = rows_before - rows_after
            
            columns_cleaned.append({
                'columna': col,
                'nulos_originales': null_count,
                'porcentaje_original': null_percentage * 100,
                'filas_eliminadas': rows_dropped
            })
            if verbose:
                print(f"🧹 {col:<25}: Filas limpiadas ({rows_dropped:,} filas eliminadas, {null_percentage*100:.1f}% nulos)")
    
    # Estadísticas finales
    final_shape = df_processed.shape
    total_nulls_remaining = df_processed.isnull().sum().sum()
    
    if verbose:
        print()
        print("📊 RESUMEN FINAL")
        print("=" * 30)
        print(f"Forma final: {final_shape[0]:,} filas x {final_shape[1]} columnas")
        print(f"Filas eliminadas: {original_shape[0] - final_shape[0]:,}")
        print(f"Columnas eliminadas: {len(columns_dropped)}")
        print(f"Columnas limpiadas: {len(columns_cleaned)}")
        print(f"Nulos restantes: {total_nulls_remaining:,}")
        
        if columns_dropped:
            print(f"\n🗑️ COLUMNAS ELIMINADAS ({len(columns_dropped)}):")
            for col_info in columns_dropped:
                print(f"   • {col_info['columna']}: {col_info['porcentaje']:.1f}% nulos")
        
        if columns_cleaned:
            print(f"\n🧹 COLUMNAS LIMPIADAS ({len(columns_cleaned)}):")
            for col_info in columns_cleaned[:5]:  # Mostrar solo las primeras 5
                print(f"   • {col_info['columna']}: {col_info['filas_eliminadas']:,} filas eliminadas")
            if len(columns_cleaned) > 5:
                print(f"   • ... y {len(columns_cleaned) - 5} columnas más")
        
        if total_nulls_remaining > 0:
            print(f"\n⚠️  Advertencia: Quedan {total_nulls_remaining:,} valores nulos")
        else:
            print(f"\n✅ Resultado: Dataset completamente limpio")


        # Calcular estadísticas
        nulls_before = data.isnull().sum().sum()
        nulls_after = df_processed.isnull().sum().sum()
        
        rows_dropped = data.shape[0] - df_processed.shape[0]
        cols_dropped = data.shape[1] - df_processed.shape[1]
        
        # Identificar qué columnas se eliminaron
        dropped_columns = set(data.columns) - set(df_processed.columns)
        
        markdown = f"""
⚙️ Configuración
- **Umbral para eliminar columna**: {threshold*100:.0f}%
- **Estrategia**: Eliminar columna si nulos ≥ {threshold*100:.0f}%, sino eliminar filas con nulos

📊 Resultados

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|---------|
| **Filas** | {data.shape[0]:,} | {df_processed.shape[0]:,} | -{rows_dropped:,} |
| **Columnas** | {data.shape[1]:,} | {df_processed.shape[1]:,} | -{cols_dropped:,} |
| **Valores nulos** | {nulls_before:,} | {nulls_after:,} | -{nulls_before - nulls_after:,} |

"""
        
        # Estado final
        if nulls_after == 0:
            markdown += "\n✅ Resultado: Dataset completamente limpio\n"
        else:
            remaining_nulls = df_processed.isnull().sum()
            remaining_nulls = remaining_nulls[remaining_nulls > 0]
            
            markdown += f"\n⚠️ Nulos Restantes: {len(remaining_nulls)} columnas\n"
            for col, count in remaining_nulls.head(5).items():
                pct = (count / len(df_processed)) * 100
                markdown += f"- **{col}**: {count:,} nulos ({pct:.1f}%)\n"

        with open('./template/report.md', 'r', encoding='utf-8') as f:
            template = f.read()

        # Reemplazar el marcador de posición en la plantilla
        pattern = r'\{proceso_nulos\}'
        result = re.sub(pattern, markdown, template)

        with open('./template/report.md', 'w', encoding='utf-8') as f:
            f.write(result)
    
    return df_processed, markdown

def outlier_processing(data, iqr_multiplier=1.5):
    """
    Detecta y elimina outliers en el dataset utilizando únicamente el método IQR.
    
    Args:
        data (pd.DataFrame): DataFrame con datos numéricos.
        iqr_multiplier (float): Multiplicador del rango intercuartílico (por defecto 1.5).
        
    Returns:
        DataFrame: DataFrame sin outliers.
        markdown: Resumen del proceso en formato Markdown.
    """
    
    if data is None or data.empty:
        return None, None
    
    # Crear copia para no modificar el original
    df_processed = data.copy()
    original_shape = df_processed.shape
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("❌ No hay columnas numéricas en el DataFrame para procesar outliers.")
    
    # Inicializar información para el markdown
    outliers_info = {
        'original_rows': original_shape[0],
        'original_cols': original_shape[1],
        'numeric_columns': len(numeric_cols),
        'iqr_multiplier': iqr_multiplier,
        'columns_analysis': {},
        'total_outliers': 0,
        'columns_with_outliers': 0
    }
    
    # Máscara global para identificar filas con outliers
    global_outliers_mask = pd.Series(False, index=df_processed.index)
    
    # Procesar cada columna numérica
    for col in numeric_cols:
        # Obtener datos válidos (sin NaN)
        col_data = df_processed[col].dropna()
        
        # Verificar que hay suficientes datos
        if len(col_data) < 3:
            outliers_info['columns_analysis'][col] = {
                'outliers_count': 0,
                'outliers_percentage': 0.0,
                'status': 'Datos insuficientes',
                'min_value': None,
                'max_value': None,
                'min_outlier': None,
                'max_outlier': None,
                'iqr_outliers': 0
            }
            continue
        
        # Calcular percentiles para IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Obtener rango completo de valores
        min_value = df_processed[col].min()
        max_value = df_processed[col].max()
        
        # Verificar que hay variabilidad en el IQR
        if IQR == 0 or np.isnan(IQR):
            outliers_info['columns_analysis'][col] = {
                'outliers_count': 0,
                'outliers_percentage': 0.0,
                'status': 'Sin variación',
                'min_value': min_value,
                'max_value': max_value,
                'min_outlier': None,
                'max_outlier': None,
                'iqr_outliers': 0
            }
            continue
        
        # Identificar outliers por IQR
        iqr_lower_bound = Q1 - iqr_multiplier * IQR
        iqr_upper_bound = Q3 + iqr_multiplier * IQR
        iqr_outliers_mask = (df_processed[col] < iqr_lower_bound) | (df_processed[col] > iqr_upper_bound)
        iqr_outliers_count = iqr_outliers_mask.sum()
        
        # Usar solo IQR para la detección
        outliers_percentage = (iqr_outliers_count / len(df_processed)) * 100
        
        # Actualizar máscara global
        global_outliers_mask |= iqr_outliers_mask
        
        # Obtener estadísticas de outliers
        if iqr_outliers_count > 0:
            outlier_values = df_processed.loc[iqr_outliers_mask, col]
            min_outlier = outlier_values.min()
            max_outlier = outlier_values.max()
            outliers_info['total_outliers'] += iqr_outliers_count
            outliers_info['columns_with_outliers'] += 1
        else:
            min_outlier = None
            max_outlier = None
        
        # Guardar información detallada de la columna
        outliers_info['columns_analysis'][col] = {
            'outliers_count': iqr_outliers_count,
            'outliers_percentage': outliers_percentage,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'iqr_lower_bound': iqr_lower_bound,
            'iqr_upper_bound': iqr_upper_bound,
            'min_value': min_value,
            'max_value': max_value,
            'min_outlier': min_outlier,
            'max_outlier': max_outlier,
            'iqr_outliers': iqr_outliers_count,
            'status': 'Procesado'
        }
    
    # Eliminar filas con outliers
    df_processed = df_processed[~global_outliers_mask]
    
    # Calcular estadísticas finales
    final_shape = df_processed.shape
    rows_removed = original_shape[0] - final_shape[0]
    removal_percentage = (rows_removed / original_shape[0]) * 100 if original_shape[0] > 0 else 0
    
    # Actualizar información final
    outliers_info.update({
        'final_rows': final_shape[0],
        'rows_removed': rows_removed,
        'removal_percentage': removal_percentage
    })
    
    # Generar markdown
    markdown = f"""
⚙️ Configuración
- **Método**: IQR (Rango Intercuartílico)
- **Multiplicador IQR**: {outliers_info['iqr_multiplier']}
- **Criterio**: Una fila se elimina si tiene outliers por IQR en cualquier columna numérica

📊 Resumen General

| Métrica | Valor |
|---------|-------|
| **Filas originales** | {outliers_info['original_rows']:,} |
| **Filas finales** | {outliers_info['final_rows']:,} |
| **Filas eliminadas** | {outliers_info['rows_removed']:,} |
| **Porcentaje eliminado** | {outliers_info['removal_percentage']:.1f}% |
| **Columnas numéricas** | {outliers_info['numeric_columns']} |
| **Columnas con outliers** | {outliers_info['columns_with_outliers']} |
| **Total outliers detectados** | {outliers_info['total_outliers']:,} |

"""

    # Añadir detalles por columna si hay outliers
    if outliers_info['columns_with_outliers'] > 0:
        markdown += """📈 Análisis Detallado por Columna

| Columna | Outliers | Porcentaje | Rango de Valores | Rango Outliers | Estado |
|---------|----------|------------|------------------|----------------|--------|
"""
        
        for col, info in outliers_info['columns_analysis'].items():
            if info['status'] == 'Procesado':
                # Rango completo de valores
                if info['min_value'] is not None and info['max_value'] is not None:
                    rango_valores = f"{info['min_value']:.2f} - {info['max_value']:.2f}"
                else:
                    rango_valores = "-"
                
                # Rango de outliers
                if info['outliers_count'] > 0 and info['min_outlier'] is not None and info['max_outlier'] is not None:
                    rango_outliers = f"{info['min_outlier']:.2f} - {info['max_outlier']:.2f}"
                    estado = "🚨 Con outliers"
                else:
                    rango_outliers = "-"
                    estado = "✅ Limpio"
                
                markdown += f"| `{col}` | {info['iqr_outliers']:,} | {info['outliers_percentage']:.1f}% | {rango_valores} | {rango_outliers} | {estado} |\n"
            else:
                # Para columnas con problemas
                if info['min_value'] is not None and info['max_value'] is not None:
                    rango_valores = f"{info['min_value']:.2f} - {info['max_value']:.2f}"
                else:
                    rango_valores = "-"
                
                markdown += f"| `{col}` | - | - | {rango_valores} | - | ⚠️ {info['status']} |\n"
        
        # Añadir interpretación de resultados        
        if outliers_info['removal_percentage'] < 5:
            markdown += "\n✅ **Impacto Bajo**: La eliminación de outliers conserva la mayoría de los datos (>95%).\n\n"
        elif outliers_info['removal_percentage'] < 15:
            markdown += "\n⚠️ **Impacto Moderado**: Se eliminó un porcentaje considerable de datos.\n\n"
        else:
            markdown += "\n🚨 **Impacto Alto**: Se eliminó una cantidad significativa de datos.\n\n"

    else:
        markdown += "✅ Resultado Excelente\n\n"
        markdown += "No se detectaron outliers en ninguna columna numérica. El dataset está limpio.\n"
    

    # Seleccionar solo columnas numéricas
    num_cols = df_processed.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        print("No se encontraron columnas numéricas.")
        return {}, ""

    # Calcular estadísticas descriptivas para datos completos
    descriptive_stats = {}
    for col in num_cols:
        base_stats = df_processed[col].describe()
        # Agregar percentiles adicionales (1%, 5%, 10%, 90%, 95%, 99%)
        p1 = df_processed[col].quantile(0.01)
        p5 = df_processed[col].quantile(0.05)
        p10 = df_processed[col].quantile(0.10)
        p90 = df_processed[col].quantile(0.90)
        p95 = df_processed[col].quantile(0.95)
        p99 = df_processed[col].quantile(0.99)
        
        # Crear diccionario extendido con los nuevos percentiles
        extended_stats = base_stats.to_dict()
        extended_stats['1%'] = p1
        extended_stats['5%'] = p5
        extended_stats['10%'] = p10
        extended_stats['90%'] = p90
        extended_stats['95%'] = p95
        extended_stats['99%'] = p99
        
        # Convertir de vuelta a Series para mantener compatibilidad
        descriptive_stats[col] = pd.Series(extended_stats)

    # Crear la figura y los subplots
    n_rows = len(num_cols)
    n_cols = 2
    figsize = (14, 5 * n_rows)
    gridspec_kw = {'width_ratios': [4, 1]}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             gridspec_kw=gridspec_kw, squeeze=False)

    fig.suptitle('Análisis Descriptivo de Columnas Numéricas (Sin Outliers)', fontsize=16, weight='bold', y=1.02)

    # Iterar sobre cada columna numérica y graficar
    for i, col in enumerate(num_cols):
        # Histograma con la distribución completa
        ax_hist_full = axes[i, 0]
        sns.histplot(data=df_processed, x=col, bins=50, ax=ax_hist_full, kde=True)
        ax_hist_full.set_xlabel("Valor")
        ax_hist_full.set_ylabel("Frecuencia")
        ax_hist_full.set_title(f"Distribución de '{col}' (Sin Outliers)")

        # Boxplot con la distribución completa
        ax_box_full = axes[i, 1]
        sns.boxplot(y=df_processed[col], ax=ax_box_full)
        ax_box_full.set_title(f"Boxplot de '{col}'")
        ax_box_full.set_ylabel("")
        ax_box_full.set_xticks([])

    # Quita borde superior y derecho
    sns.despine()

    # Ajuste del diseño
    plt.tight_layout()
    
    # Descomenta la siguiente línea para guardar la imagen
    plt.savefig('./docs/distribucion_numericas_sin_outliers.png', dpi=300, bbox_inches='tight')

    markdown += "### Distribución de los datos sin outlliers:\n\n![Distribución de Columnas Numéricas](../docs/distribucion_numericas_sin_outliers.png)\n\n"

    with open('./template/report.md', 'r', encoding='utf-8') as f:
            template = f.read()

    # Reemplazar el marcador de posición en la plantilla
    pattern = r'\{proceso_outliers\}'
    result = re.sub(pattern, markdown, template)

    with open('./template/report.md', 'w', encoding='utf-8') as f:
        f.write(result)
    
    return df_processed, markdown

def visualize_correlations(data, target_column=None, figsize=(40, 30), save_path=None):
    """
    Visualiza correlaciones entre variables numéricas y categóricas de diferentes formas.
    
    Args:
        data (pd.DataFrame): Dataset a analizar
        target_column (str): Columna objetivo para análisis especial (opcional)
        figsize (tuple): Tamaño de la figura
        save_path (str): Ruta para guardar la imagen (opcional)
    
    Returns:
        dict: Diccionario con matrices de correlación y estadísticas
    """
    
    if data is None or data.empty:
        print("❌ Dataset vacío o None")
        return None
    
    # Identificar tipos de variables
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filtrar columnas con demasiados valores únicos para categóricas
    categorical_cols = [col for col in categorical_cols if data[col].nunique() <= 20]
    
    print(f"📊 Variables numéricas: {len(numeric_cols)}")
    print(f"📝 Variables categóricas: {len(categorical_cols)}")
    
    if len(numeric_cols) < 2 and len(categorical_cols) < 2:
        print("❌ Insuficientes variables para análisis de correlación")
        return None
    
    # Crear figura con subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[2, 2, 1], 
                         hspace=0.3, wspace=0.3)
    
    results = {}
    
    # 1. Correlación entre variables numéricas (Pearson)
    if len(numeric_cols) >= 2:
        ax1 = fig.add_subplot(gs[0, 0])
        numeric_corr = data[numeric_cols].corr()
        
        mask = np.triu(np.ones_like(numeric_corr, dtype=bool))
        sns.heatmap(numeric_corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Correlación Pearson\n(Variables Numéricas)', fontsize=12, weight='bold')
        results['numeric_correlation'] = numeric_corr
    
    # 2. Correlación Spearman (no paramétrica)
    if len(numeric_cols) >= 2:
        ax2 = fig.add_subplot(gs[0, 1])
        spearman_corr = data[numeric_cols].corr(method='spearman')
        
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax2)
        ax2.set_title('Correlación Spearman\n(Variables Numéricas)', fontsize=12, weight='bold')
        results['spearman_correlation'] = spearman_corr
    
    # 3. Matriz de asociación categórica (Cramér's V)
    if len(categorical_cols) >= 2:
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Calcular Cramér's V
        cols = categorical_cols
        n = len(cols)
        cramers_v = np.zeros((n, n))
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i == j:
                    cramers_v[i, j] = 1.0
                else:
                    try:
                        contingency_table = pd.crosstab(data[col1], data[col2])
                        chi2, _, _, _ = chi2_contingency(contingency_table)
                        n_obs = contingency_table.sum().sum()
                        cramers_v[i, j] = np.sqrt(chi2 / (n_obs * (min(contingency_table.shape) - 1)))
                    except:
                        cramers_v[i, j] = 0
        
        cramers_v_matrix = pd.DataFrame(cramers_v, index=cols, columns=cols)
        
        sns.heatmap(cramers_v_matrix, annot=True, cmap='viridis', center=0.5,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax3)
        ax3.set_title("Asociación Cramér's V\n(Variables Categóricas)", fontsize=12, weight='bold')
        results['cramers_v_matrix'] = cramers_v_matrix
    
    # 4. Análisis numérico-categórico (ANOVA F-statistic)
    if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Calcular matriz ANOVA
        f_stats = np.zeros((len(numeric_cols), len(categorical_cols)))
        
        for i, num_col in enumerate(numeric_cols):
            for j, cat_col in enumerate(categorical_cols):
                try:
                    groups = [group[num_col].dropna() for name, group in data.groupby(cat_col)]
                    groups = [g for g in groups if len(g) >= 2]
                    
                    if len(groups) >= 2:
                        f_stat, _ = stats.f_oneway(*groups)
                        f_stats[i, j] = f_stat if not np.isnan(f_stat) else 0
                    else:
                        f_stats[i, j] = 0
                except:
                    f_stats[i, j] = 0
        
        anova_matrix = pd.DataFrame(f_stats, index=numeric_cols, columns=categorical_cols)
        
        sns.heatmap(anova_matrix, annot=True, cmap='plasma', 
                   fmt='.2f', cbar_kws={"shrink": .8}, ax=ax4)
        ax4.set_title('Estadístico F (ANOVA)\n(Numérica vs Categórica)', fontsize=12, weight='bold')
        ax4.set_xlabel('Variables Categóricas')
        ax4.set_ylabel('Variables Numéricas')
        results['anova_matrix'] = anova_matrix
    
    # 5. Análisis del target (si se especifica)
    if target_column and target_column in data.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Analizar correlaciones con target
        variables = []
        correlations = []
        types = []
        
        # Para variables numéricas
        for col in numeric_cols:
            if col != target_column:
                try:
                    if data[target_column].dtype in ['int64', 'float64']:
                        corr, _ = pearsonr(data[col].dropna(), data[target_column].dropna())
                        variables.append(col)
                        correlations.append(abs(corr))
                        types.append('Numérica')
                    else:
                        groups = [group[col].dropna() for name, group in data.groupby(target_column)]
                        if len(groups) >= 2:
                            f_stat, _ = stats.f_oneway(*groups)
                            variables.append(col)
                            correlations.append(f_stat / 100)
                            types.append('Numérica')
                except:
                    continue
        
        # Para variables categóricas
        for col in categorical_cols:
            if col != target_column:
                try:
                    contingency_table = pd.crosstab(data[col], data[target_column])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    n_obs = contingency_table.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n_obs * (min(contingency_table.shape) - 1)))
                    variables.append(col)
                    correlations.append(cramers_v)
                    types.append('Categórica')
                except:
                    continue
        
        # Ordenar por fuerza de asociación
        if variables:
            sorted_indices = np.argsort(correlations)[::-1]
            
            target_correlations = {
                'variables': [variables[i] for i in sorted_indices],
                'correlations': [correlations[i] for i in sorted_indices],
                'types': [types[i] for i in sorted_indices]
            }
            
            # Visualizar
            target_df = pd.DataFrame(target_correlations)
            colors = ['#1f77b4' if t == 'Numérica' else '#ff7f0e' for t in target_df['types']]
            bars = ax5.barh(range(len(target_df)), target_df['correlations'], color=colors)
            ax5.set_yticks(range(len(target_df)))
            ax5.set_yticklabels(target_df['variables'], fontsize=10)
            ax5.set_xlabel('Fuerza de Asociación')
            ax5.set_title(f'Asociación con Target: {target_column}', fontsize=12, weight='bold')
            ax5.grid(axis='x', alpha=0.3)
            
            # Agregar valores en las barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            results['target_analysis'] = target_correlations
    
    # 6. Top correlaciones
    ax6 = fig.add_subplot(gs[1, 2])
    if len(numeric_cols) >= 2:
        corr_matrix = data[numeric_cols].corr()
        
        # Extraer correlaciones únicas
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    correlations.append((var1, var2, corr_val))
        
        # Ordenar por valor absoluto
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        top_corr = correlations[:10]
        
        if top_corr:
            y_pos = np.arange(len(top_corr))
            corr_values = [abs(corr) for _, _, corr in top_corr]
            labels = [f"{var1} - {var2}" for var1, var2, _ in top_corr]
            
            bars = ax6.barh(y_pos, corr_values, color='skyblue')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(labels, fontsize=9)
            ax6.set_xlabel('|Correlación|')
            ax6.set_title('Top Correlaciones\n(Valor Absoluto)', fontsize=12, weight='bold')
            
            # Agregar valores
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax6.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            results['top_correlations'] = top_corr
    
    # 7. Resumen estadístico
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Crear texto resumen
    summary = f"""
📊 RESUMEN DEL ANÁLISIS DE CORRELACIONES

🔢 Variables Numéricas: {len(numeric_cols)}
📝 Variables Categóricas: {len(categorical_cols)}
📋 Total de Variables: {len(data.columns)}
🗂️ Observaciones: {len(data):,}

"""
    
    if 'numeric_correlation' in results:
        max_corr = results['numeric_correlation'].abs().values[np.triu_indices_from(results['numeric_correlation'].values, k=1)].max()
        summary += f"🔗 Correlación Máxima (Pearson): {max_corr:.3f}\n"
    
    if 'cramers_v_matrix' in results:
        max_cramers = results['cramers_v_matrix'].values[np.triu_indices_from(results['cramers_v_matrix'].values, k=1)].max()
        summary += f"🔗 Asociación Máxima (Cramér's V): {max_cramers:.3f}\n"
    
    if 'top_correlations' in results and results['top_correlations']:
        top_pair = results['top_correlations'][0]
        summary += f"🏆 Par más correlacionado: {top_pair[0]} - {top_pair[1]} ({top_pair[2]:.3f})"
    
    ax7.text(0.05, 0.95, summary, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Análisis Completo de Correlaciones y Asociaciones', 
                fontsize=16, weight='bold', y=0.98)
    
    # Guardar si se especifica ruta
    plt.savefig('./docs/analisis_correlaciones.png', dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    # plt.show()
    
    return results

