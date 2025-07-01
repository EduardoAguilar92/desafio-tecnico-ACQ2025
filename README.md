## 1. Configuración del Entorno (revisar al final)

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/EduardoAguilar92/desafio-tecnico-ACQ2025.git
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
1. Colocar el archivo `new_items_dataset.csv` en la ruta `./data/raw/`
2. En la terminal, correr el siguiente comando: `python main.py --train=True` 
   - Esto iniciará el entrenamiento del modelo, además de generar un reporte de la exploración de los datos y los resultados obtenidos.
   - Tambien se puede ejecutar el comando: `python main.py --train=False` para generar el reporte teniendo ya los archivos `.pkl` generados.
3. El reporte final se encuentra en la ruta `./template/report.md`






