## 1. Configuración del Entorno (revisar al final)

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/desafio-tecnico-ACQ2025.git
    cd desafio-tecnico-ACQ2025
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows: .venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Estructura del Repositorio

```
.
├── data/
│   ├── raw/
│   │   └── new_items_dataset.csv
│   ├── processed/
│   │   └── new_items_dataset_processed.csv
│   └── results/
│       └── predicciones.csv
├── docs/
│   └── graficos
├── notebooks/
│   └── 02_EDA.ipynb
├── models/
│   └── training_session_2025MMDD_HHMMSS/
├── notebooks/
│   └── 02_EDA.ipynb
├── templates/
|   ├── report.md
│   └── report_template.md
├── utils/
│   └── statistics.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## 6. Instalación y Uso

Para replicar el entorno y ejecutar el proyecto, siga los siguientes pasos:



## Objetivos:
- Análisis exploratorio completo del dataset de MercadoLibre
- Feature Engineering avanzado para crear variables predictivas
- Modelado predictivo con múltiples algoritmos de Machine Learning
- Generación de insights accionables para Marketing y Negocio
- Pipeline de producción escalable y mantenible

## Análisis Exploratorio de Datos (EDA):
El dataset `new_items_dataset.csv` tiene 100.000 registros de ítems extraídos del marketplace en MercadoLibre, caracterizados a través de 26 diferentes columnas, de las cuales 20 son `categóricas` y 6 son `numéricas`.



