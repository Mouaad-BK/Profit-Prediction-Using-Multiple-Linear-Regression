import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import category_encoders as ce
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variables globales
data = None
current_data = None
trained_model = None
global_encoder = None
global_scaler = None
global_pca = None
pca_features = []       # Colonnes utilisées pour l'ACP
model_features = []     # Colonnes utilisées pour l'entraînement du modèle
X_test = None
y_test = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def get_data_stats(df):
    """Extrait les statistiques du DataFrame, détermine
    les variables indépendantes et la variable cible (si 'Profit' existe),
    et affiche l'intégralité du DataFrame."""
    if df is None or df.empty:
        return {
            'instance_count': 0,
            'attribute_count': 0,
            'data_preview': '<p>Aucun fichier chargé</p>',
            'missing_rows_count': 0,
            'missing_columns': [],
            'corr_image': None,
            'pca_image': None,
            'plot_image': None,
            'independent_vars': "Non défini",
            'target_var': "Non défini"
        }
    else:
        instance_count = df.shape[0]
        columns = list(df.columns)
        if 'Profit' in columns:
            target_var = 'Profit'
            independent_vars = ", ".join([col for col in columns if col != 'Profit'])
        else:
            target_var = "Non défini"
            independent_vars = ", ".join(columns)
        return {
            'instance_count': instance_count,
            'attribute_count': df.shape[1],
            'data_preview': df.to_html(classes='data-table', index=False),
            'missing_rows_count': df.isnull().any(axis=1).sum(),
            'missing_columns': df.columns[df.isnull().any()].tolist(),
            'corr_image': None,
            'pca_image': None,
            'plot_image': None,
            'independent_vars': independent_vars,
            'target_var': target_var
        }

@app.route('/')
def home():
    stats = get_data_stats(current_data)
    return render_template('index.html', **stats)

@app.route('/upload_file', methods=['POST'])
def upload_file():
    global data, current_data
    if 'file' not in request.files:
        stats = get_data_stats(current_data)
        return render_template('index.html', upload_message="Aucun fichier sélectionné !", **stats)
    file = request.files['file']
    if file.filename == '':
        stats = get_data_stats(current_data)
        return render_template('index.html', upload_message="Aucun fichier sélectionné !", **stats)
    if file and allowed_file(file.filename):
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            data = df.copy()
            current_data = df.copy()
            stats = get_data_stats(current_data)
            return render_template('index.html', upload_message="Fichier chargé avec succès !", **stats)
        except Exception as e:
            stats = get_data_stats(current_data)
            return render_template('index.html', upload_message="Erreur lors du chargement du fichier: " + str(e), **stats)
    else:
        stats = get_data_stats(current_data)
        return render_template('index.html', upload_message="Format de fichier non supporté. Seuls les fichiers CSV sont autorisés.", **stats)

# Décorateur pour vérifier qu'un fichier a été chargé
def require_data(func):
    def wrapper(*args, **kwargs):
        if current_data is None or current_data.empty:
            stats = get_data_stats(current_data)
            return render_template('index.html', upload_message="Veuillez charger un fichier CSV pour commencer !", **stats)
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/clean', methods=['POST'])
@require_data
def clean_data():
    global current_data
    strategy = request.form.get('missingStrategy')
    if strategy in ['mean', 'median']:
        imputer = SimpleImputer(strategy=strategy)
        numeric_cols = current_data.select_dtypes(include=['number']).columns
        current_data[numeric_cols] = imputer.fit_transform(current_data[numeric_cols])
    elif strategy == 'drop':
        current_data = current_data.dropna()
    stats = get_data_stats(current_data)
    return render_template('index.html', **stats)

@app.route('/encode_normalize', methods=['POST'])
@require_data
def encode_normalize():
    global current_data, global_encoder, global_scaler
    # --- Encodage ---
    encoding_method = request.form.get('encodingMethod')
    if encoding_method == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(current_data[['State']]).astype(np.int32)
        encoded_cols = encoder.get_feature_names_out(['State'])
        current_data = pd.concat([current_data.drop('State', axis=1),
                                  pd.DataFrame(encoded, columns=encoded_cols, index=current_data.index)],
                                  axis=1)
        global_encoder = encoder
    elif encoding_method == 'label':
        encoder = LabelEncoder()
        current_data['State'] = encoder.fit_transform(current_data['State']).astype(np.int32)
        global_encoder = encoder
    elif encoding_method == 'binary':
        encoder = ce.BinaryEncoder(cols=['State'])
        current_data = encoder.fit_transform(current_data)
        state_cols = [col for col in current_data.columns if col.startswith('State_')]
        current_data[state_cols] = current_data[state_cols].astype(np.int32)
        global_encoder = encoder

    # --- Normalisation ---
    normalization_method = request.form.get('normalizationMethod')
    exclude_cols = ['Profit']
    if global_encoder:
        if isinstance(global_encoder, OneHotEncoder) or 'BinaryEncoder' in str(type(global_encoder)):
            exclude_cols += [col for col in current_data.columns if col.startswith("State_")]
        elif isinstance(global_encoder, LabelEncoder):
            exclude_cols.append('State')
    numeric_cols = current_data.select_dtypes(include=['number']).columns.difference(exclude_cols).tolist()
    if normalization_method == 'standard':
        scaler = StandardScaler()
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()
    elif normalization_method == 'maxabs':
        scaler = MaxAbsScaler()
    elif normalization_method == 'robuste':
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler and numeric_cols:
        current_data[numeric_cols] = scaler.fit_transform(current_data[numeric_cols]).astype(np.float32)
        global_scaler = scaler

    stats = get_data_stats(current_data)
    return render_template('index.html', **stats)

@app.route('/generate_corr', methods=['POST'])
@require_data
def generate_corr():
    global current_data
    excluded_columns = ['Profit'] + [col for col in current_data.columns if 'State' in col]
    features = [col for col in current_data.columns if col not in excluded_columns]
    if len(features) == 0:
        stats = get_data_stats(current_data)
        return render_template('index.html', corr_image=None, **stats)
    corr_matrix = current_data[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img_path = f'static/corr_matrix_{timestamp}.png'
    plt.savefig(img_path)
    plt.close()
    stats = get_data_stats(current_data)
    stats['corr_image'] = img_path
    return render_template('index.html', **stats)

@app.route('/apply_pca', methods=['POST'])
@require_data
def apply_pca():
    global current_data, global_pca, pca_features
    excluded = ['Profit'] + [col for col in current_data.columns if 'State' in col]
    features = [col for col in current_data.columns if col not in excluded]
    if len(features) < 2:
        return "Pas assez de caractéristiques pour l'ACP.", 400
    pca_features = features
    pca_data = current_data[pca_features]
    if pca_data.isnull().any().any():
        pca_data = pca_data.fillna(pca_data.mean())
    n_components = int(request.form.get('pcaComponents', 2))
    global_pca = PCA(n_components=n_components)
    pca_result = global_pca.fit_transform(pca_data)
    current_data = current_data.drop(columns=pca_features)
    for i in range(n_components):
        current_data[f'PC{i+1}'] = pca_result[:, i]
    plt.figure(figsize=(8, 6))
    if n_components >= 2:
        sns.scatterplot(x=current_data['PC1'], y=current_data['PC2'])
    else:
        sns.histplot(current_data['PC1'], kde=True)
    img_path = f'static/pca_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    plt.savefig(img_path)
    plt.close()
    stats = get_data_stats(current_data)
    stats['pca_image'] = img_path
    return render_template('index.html', **stats)

@app.route('/train_model', methods=['POST'])
@require_data
def train_model():
    global trained_model, X_test, y_test, model_features, current_data
    if 'Profit' not in current_data.columns:
        stats = get_data_stats(current_data)
        return render_template('index.html', train_message="❌ La colonne 'Profit' est manquante !", **stats)
    X = current_data.drop('Profit', axis=1)
    y = current_data['Profit']
    model_features = X.columns.tolist()
    train_split = int(request.form.get('trainSplit', 80))
    test_size = (100 - train_split) / 100.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    trained_model = LinearRegression().fit(X_train, y_train)
    stats = get_data_stats(current_data)
    return render_template('index.html', train_message=f"✅ Modèle entraîné ({train_split}%) !", **stats)

@app.route('/test_model', methods=['POST'])
@require_data
def test_model():
    global trained_model, X_test, y_test, current_data
    if not trained_model:
        stats = get_data_stats(current_data)
        return render_template('index.html', test_message="❌ Entraînez le modèle d'abord !", **stats)
    y_pred = trained_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Prédictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Valeurs idéales')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.title('Comparaison Réelles vs Réelles')
    plt.legend()
    plot_path = f'static/pred_plot_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    plt.savefig(plot_path)
    plt.close()
    stats = get_data_stats(current_data)
    stats.update({'r2': r2, 'mse': mse, 'plot_image': plot_path, 'test_message': None})
    return render_template('index.html', **stats)

@app.route('/predict', methods=['POST'])
@require_data
def predict():
    global trained_model, model_features, global_encoder, global_scaler, global_pca, pca_features, current_data
    if trained_model is None:
        stats = get_data_stats(current_data)
        return render_template('index.html', predicted_profit="❌ Entraînez d'abord le modèle !", **stats)
    try:
        # Création du DataFrame d'entrée
        input_df = pd.DataFrame([{
            'R&D Spend': float(request.form['rdSpend']),
            'Administration': float(request.form['administration']),
            'Marketing Spend': float(request.form['marketingSpend']),
            'State': request.form['state']
        }])
        # Encodage
        if global_encoder:
            if isinstance(global_encoder, OneHotEncoder):
                encoded = global_encoder.transform(input_df[['State']]).astype(np.int32)
                encoded_cols = global_encoder.get_feature_names_out(['State'])
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=input_df.index)
                input_df = pd.concat([input_df.drop('State', axis=1), encoded_df], axis=1)
            elif isinstance(global_encoder, LabelEncoder):
                try:
                    input_df['State'] = global_encoder.transform(input_df['State']).astype(np.int32)
                except ValueError:
                    input_df['State'] = -1
            elif 'BinaryEncoder' in str(type(global_encoder)):
                input_df = global_encoder.transform(input_df).astype(np.int32)
                state_cols = [col for col in input_df.columns if col.startswith('State_')]
                input_df[state_cols] = input_df[state_cols].astype(np.int32)
        # Normalisation
        if global_scaler:
            exclude_cols = []
            if global_encoder:
                if isinstance(global_encoder, OneHotEncoder) or 'BinaryEncoder' in str(type(global_encoder)):
                    exclude_cols = [col for col in input_df.columns if col.startswith("State_")]
                elif isinstance(global_encoder, LabelEncoder):
                    exclude_cols = ['State']
            num_cols = input_df.select_dtypes(include=['number']).columns.difference(['Profit'] + list(exclude_cols))
            if len(num_cols) > 0:
                input_df[num_cols] = global_scaler.transform(input_df[num_cols])
        # Application de l'ACP
        if global_pca and pca_features:
            pca_input = input_df[pca_features].fillna(input_df[pca_features].mean())
            pca_result = global_pca.transform(pca_input)
            input_df = input_df.drop(columns=pca_features)
            for i in range(global_pca.n_components_):
                input_df[f'PC{i+1}'] = pca_result[:, i]
        # Réalignement des colonnes pour correspondre aux features du modèle
        input_df = input_df.reindex(columns=model_features, fill_value=0)
        # Prédiction finale
        prediction = trained_model.predict(input_df)
        predicted_profit = f"${prediction[0]:.2f}"
        stats = get_data_stats(current_data)
        return render_template('index.html', predicted_profit=predicted_profit, **stats)
    except Exception as e:
        stats = get_data_stats(current_data)
        return render_template('index.html', predicted_profit=f"❌ Erreur: {str(e)}", **stats)

@app.route('/reset_data', methods=['POST'])
def reset_data():
    global data, current_data, trained_model, global_encoder, global_scaler, global_pca, pca_features
    data = None
    current_data = None
    trained_model = None
    global_encoder = None
    global_scaler = None
    global_pca = None
    pca_features = []
    stats = get_data_stats(current_data)
    return render_template('index.html', upload_message="✅ Données réinitialisées !", **stats)

if __name__ == '__main__':
    app.run(debug=True)