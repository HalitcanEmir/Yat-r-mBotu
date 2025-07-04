import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from xgboost import XGBRegressor
    xgb_available = True
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    tf_available = True
except ImportError:
    xgb_available = False
    tf_available = False

# Örnek fonksiyon: DataFrame'den fiyat tahmini için regresyon modeli eğitimi

def train_price_regression(
    df,
    feature_cols=None,
    target_col='Close',
    test_size=0.2,
    random_state=42,
    model_type='linear',
    plot=True,
    save_model_path=None
):
    """
    Gelişmiş regresyon pipeline.
    model_type: 'linear', 'ridge', 'lasso', 'rf', 'xgb'
    plot: True ise tahmin ve gerçek değer grafiği çizer
    save_model_path: modeli dosyaya kaydeder (opsiyonel)
    """
    # Target leakage'i önlemek için geçmiş gün verileriyle tahmin (lag)
    df = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [target_col, 'Date', 'date', 'Ticker', 'ticker']]
    # Lag feature: hedefi bir gün ileri kaydır
    df['target_next'] = df[target_col].shift(-1)
    data = df.dropna(subset=feature_cols + ['target_next'])
    X = data[feature_cols]
    y = data['target_next']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
    # Model seçimi
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'lasso':
        model = Lasso()
    elif model_type == 'rf':
        model = RandomForestRegressor(random_state=random_state)
    elif model_type == 'xgb' and xgb_available:
        model = XGBRegressor(random_state=random_state)
    else:
        raise ValueError(f"Desteklenmeyen model_type: {model_type}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f} | Test R2: {r2:.4f}")
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(y_test.values, label='Gerçek')
        plt.plot(y_pred, label='Tahmin')
        plt.title('Test Seti: Gerçek vs Tahmin')
        plt.legend()
        plt.show()
        # Özellik önemleri (varsa)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(8,4))
            plt.bar(X_train.columns, model.feature_importances_)
            plt.title('Özellik Önemleri')
            plt.show()
    if save_model_path:
        joblib.dump(model, save_model_path)
    return model, X_test, y_test, y_pred

def train_lstm_price_forecast(df, feature_cols=None, target_col='Close', window_size=30, epochs=20, batch_size=32, plot=True):
    """
    LSTM ile zaman serisi fiyat tahmini pipeline'ı.
    """
    if not tf_available:
        print("TensorFlow/Keras yüklü değil, LSTM pipeline atlanıyor.")
        return None, None, None
    if feature_cols is None:
        feature_cols = ['Close', 'RSI', 'MACD', 'MA50', 'MA200', 'Volume']
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols + [target_col]])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, :-1])
        y.append(scaled[i, -1])
    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        print("Yeterli veri yok, LSTM eğitilemiyor.")
        return None, None, None
    # Train/test split (zaman sıralı)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    y_pred = model.predict(X_test)
    # Ters ölçekleme
    y_test_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), X.shape[2])), y_test.reshape(-1,1)], axis=1))[:,-1]
    y_pred_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred), X.shape[2])), y_pred], axis=1))[:,-1]
    # Grafik
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(y_test_inv, label='Gerçek')
        plt.plot(y_pred_inv, label='Tahmin')
        plt.title('LSTM Fiyat Tahmini')
        plt.legend()
        plt.show()
    return model, y_test_inv, y_pred_inv

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_lstm_pytorch_price_forecast(df, feature_cols=None, target_col='Close', window_size=30, epochs=20, batch_size=32, plot=True, device=None):
    """
    PyTorch ile LSTM zaman serisi fiyat tahmini pipeline'ı.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if feature_cols is None:
        feature_cols = ['Close', 'RSI', 'MACD', 'MA50', 'MA200', 'Volume']
    df = df.dropna(subset=feature_cols + [target_col]).copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols + [target_col]])
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, :-1])
        y.append(scaled[i, -1])
    X, y = np.array(X), np.array(y)
    if len(X) < 10:
        print("Yeterli veri yok, LSTM eğitilemiyor.")
        return None, None, None
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1).to(device)
    model = LSTMRegressor(input_size=X.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test)
                val_loss = criterion(val_pred, y_test).item()
            print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f}")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    # Ters ölçekleme
    y_test_np = y_test.cpu().numpy()
    y_test_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test_np), X.shape[2])), y_test_np], axis=1))[:,-1]
    y_pred_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred), X.shape[2])), y_pred], axis=1))[:,-1]
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(y_test_inv, label='Gerçek')
        plt.plot(y_pred_inv, label='Tahmin')
        plt.title('PyTorch LSTM Fiyat Tahmini')
        plt.legend()
        plt.show()
    return model, y_test_inv, y_pred_inv

def detect_anomalies_isolation_forest(df, feature_cols=None, contamination=0.01, plot=True):
    """
    Isolation Forest ile fiyat/gösterge anomali tespiti.
    feature_cols: Hangi sütunlarda anomali aranacak (varsayılan: Close, RSI, MACD, MA50, MA200, Volume)
    contamination: Anomali oranı (varsayılan: 0.01)
    plot: True ise anomali noktalarını grafikle gösterir
    """
    if feature_cols is None:
        feature_cols = ['Close', 'RSI', 'MACD', 'MA50', 'MA200', 'Volume']
    df = df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].values
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly_score'] = model.decision_function(X)
    df['anomaly'] = model.predict(X)
    # -1: anomali, 1: normal
    anomalies = df[df['anomaly'] == -1]
    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(df.index, df['Close'], label='Close')
        plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomali', marker='x')
        plt.title('Isolation Forest ile Anomali Tespiti')
        plt.legend()
        plt.show()
    return df, anomalies

def detect_anomalies_autoencoder(df, feature_cols=None, epochs=30, batch_size=32, threshold=None, plot=True, device=None):
    """
    PyTorch Autoencoder ile anomali tespiti.
    feature_cols: Kullanılacak sütunlar (varsayılan: Close, RSI, MACD, MA50, MA200, Volume)
    threshold: Anomali eşiği (None ise otomatik belirlenir)
    plot: True ise anomali noktalarını grafikle gösterir
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if feature_cols is None:
        feature_cols = ['Close', 'RSI', 'MACD', 'MA50', 'MA200', 'Volume']
    df = df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].values.astype(np.float32)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    class Autoencoder(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 4)
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 8), nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, n_features)
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
    model = Autoencoder(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch_x = batch[0]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor).cpu().numpy()
    mse = np.mean((X_scaled - recon) ** 2, axis=1)
    if threshold is None:
        threshold = np.percentile(mse, 99)  # En yüksek %1 anomali
    df['anomaly_score_ae'] = mse
    df['anomaly_ae'] = (mse > threshold).astype(int)
    anomalies = df[df['anomaly_ae'] == 1]
    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(df.index, df['Close'], label='Close')
        plt.scatter(anomalies.index, anomalies['Close'], color='orange', label='Autoencoder Anomali', marker='x')
        plt.title('Autoencoder ile Anomali Tespiti')
        plt.legend()
        plt.show()
    return df, anomalies

def train_strategy_selector(X, y, model_type='rf', test_size=0.2, random_state=42):
    """
    ML tabanlı strateji seçici. X: piyasa göstergeleri, y: strateji etiketi (örn. 'trend', 'mean', 'volatility')
    model_type: 'rf' (RandomForest), ileride 'xgb', 'svc' eklenebilir.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise NotImplementedError('Sadece RandomForest destekleniyor.')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Strateji Seçici Test Sonuçları:')
    print(classification_report(y_test, y_pred))
    print('Doğruluk:', accuracy_score(y_test, y_pred))
    return model

def predict_strategy(model, X):
    """
    Eğitilmiş model ile strateji tahmini yapar.
    X: piyasa göstergeleri DataFrame veya array
    """
    return model.predict(X)

def monte_carlo_portfolio_simulation(price_df, weights, n_days=30, n_sim=1000, initial_value=100000, plot=True):
    """
    price_df: DataFrame, her sütun bir hisse, satırlar kapanış fiyatı
    weights: dict, portföydeki ağırlıklar (ör. {'AAPL': 0.2, ...})
    n_days: Simülasyon süresi (gün)
    n_sim: Simülasyon sayısı
    initial_value: Başlangıç portföy değeri
    """
    returns = price_df.pct_change().dropna()
    mu = returns.mean().values
    sigma = returns.std().values
    tickers = list(price_df.columns)
    weights_arr = np.array([weights.get(t, 0) for t in tickers])
    sim_results = np.zeros((n_sim, n_days))
    for i in range(n_sim):
        prices = np.ones(len(tickers)) * initial_value
        for d in range(n_days):
            rand_returns = np.random.normal(mu, sigma)
            prices = prices * (1 + rand_returns)
            sim_results[i, d] = np.dot(prices, weights_arr) / len(tickers)
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(sim_results.T, color='grey', alpha=0.1)
        plt.title('Monte Carlo Portföy Simülasyonu')
        plt.xlabel('Gün')
        plt.ylabel('Portföy Değeri')
        plt.show()
        plt.hist(sim_results[:,-1], bins=50, alpha=0.7)
        plt.title('Son Portföy Değeri Dağılımı')
        plt.xlabel('Değer')
        plt.ylabel('Frekans')
        plt.show()
        var_5 = np.percentile(sim_results[:,-1], 5)
        print(f'%5 Value at Risk (VaR): {var_5:.2f}')
        print(f'Beklenen Son Değer: {np.mean(sim_results[:,-1]):.2f}')
    return sim_results

# Kullanım örneği:
# from ml_models import train_price_regression
# model, X_test, y_test, y_pred = train_price_regression(df, model_type='rf') 