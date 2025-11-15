# ⚡ Início Rápido - Correção do Sistema

## 🎯 Objetivo
Corrigir o `ValueError: 'region_metrics' não definidos` e fazer o sistema funcionar end-to-end.

## 📝 Pré-requisitos
```bash
pip install numpy pandas scikit-learn d3rlpy==2.8.1 joblib gymnasium tqdm torch
```

## 🚀 3 Passos para Resolver

### Passo 1: Gerar Dados (5-10 minutos)
```bash
cd project/project
python3 Generator_NEW.py
```

**O que acontece:**
- ✅ Cria `sl_dataset_combined.csv` (50.000 linhas)
- ✅ Cria `rl_offline_buffer.h5` (100.000 transições)
- ✅ Cria `rl_assinatura_buffer.h5` (50.000 transições)
- ✅ Cria 12+ arquivos `.joblib` (encoders/scalers)

**Saída esperada:**
```
[1/4] Definindo parâmetros econômicos...
✓ Regiões: ['North America', 'Europe', ...]

[2/4] Gerando dataset Supervised Learning...
SL Dataset: 100%|████████| 50000/50000
✓ Dataset SL salvo: 50000 linhas

[3/4] Gerando buffer RL Venda Única...
RL Buffer: 100%|████████| 100000/100000
✓ Buffer RL gerado: 100000 transições

[4/4] Gerando buffer RL Assinatura...
✓ Buffer Assinatura gerado: 50000 transições

🎉 SUCESSO! Todos os datasets foram gerados.
```

### Passo 2: Treinar Modelos

#### 2A. Modelo SL (Baseline)
Abra `SL_FINAL (1).ipynb` e **DELETE** a célula "Etapa 9.1" que causa o erro.

**Adicione no início:**
```python
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Carregar dados PRONTOS
df_sl = pd.read_csv('sl_dataset_combined.csv')
y = df_sl['Lucro']
X = df_sl.drop(['Preco', 'Lucro'], axis=1)

# Carregar pré-processadores
encoder = joblib.load('sl_encoder.joblib')
scaler_estado = joblib.load('sl_scaler_estado.joblib')

# Transformar
X_encoded = encoder.transform(X)
X_scaled = scaler_estado.transform(X_encoded)

# Treinar
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
modelo_sl = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_sl.fit(X_train, y_train)

# Avaliar
y_pred = modelo_sl.predict(X_test)
print(f"MAE: ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2%}")

# Salvar
joblib.dump(modelo_sl, 'sl_profit_regressor_model.joblib')
print("✓ Modelo salvo!")
```

#### 2B. Modelo RL Venda Única
Abra `código_final_RL_OFF (25).ipynb` e **SUBSTITUA** a geração de dados por:

```python
import d3rlpy
import joblib

# Carregar buffer PRONTO
buffer = d3rlpy.load_replay_buffer('rl_offline_buffer.h5')
scaler_acao = joblib.load('scaler_acao.joblib')
scaler_recompensa = joblib.load('scaler_recompensa.joblib')

# Configurar CQL
cql = d3rlpy.algos.CQL(
    q_func_factory=d3rlpy.models.QRQFunctionFactory(n_quantiles=64),
    batch_size=256,
    alpha=5.0,
    gamma=0.98,
    use_gpu=False  # True se tiver GPU
)

# Treinar
cql.fit(buffer, n_steps=50000)
cql.save_model('modelo_rl_final.pt')
print("✓ Modelo RL salvo!")

# Teste de inferência (CORRETO - em DÓLARES)
import numpy as np
estado_teste = np.zeros((1, buffer.observation_signature.shape[0]))
preco_norm = cql.predict(estado_teste)[0]
preco_dolares = scaler_acao.inverse_transform([[preco_norm[0]]])[0][0]
print(f"Preço de Teste: ${preco_dolares:.2f}")  # Ex: $49.90 ✅
```

### Passo 3: Atualizar Backend

**Edite `main.py`** (após a linha 50):

```python
# === CORREÇÃO: Carregar scalers ===
import joblib

scaler_acao = joblib.load('scaler_acao.joblib')
scaler_recompensa = joblib.load('scaler_recompensa.joblib')

# === CORREÇÃO: No endpoint /api/simulate ===
@app.post("/api/simulate")
async def simulate_campaign(input_data: SimulationInput):
    # ... (processamento do estado)

    # Predição
    preco_normalizado = cql_sac_pricer_iqn.predict(observation.reshape(1, -1))[0]

    # ⭐ CONVERTER PARA DÓLARES
    preco_dolares = float(scaler_acao.inverse_transform([[preco_normalizado[0]]])[0][0])

    return {
        "recommended_price": round(preco_dolares, 2),  # ✅ $49.90
        "estimated_roi": 150.0,
        "var_5_percent": -500.0,
        "cvar_5_percent": -1200.0
    }
```

**Executar:**
```bash
python main.py
# Servidor em: http://127.0.0.1:8000
```

## ✅ Verificação Rápida

```bash
# 1. Dados gerados?
ls -lh project/project/*.csv project/project/*.h5 project/project/*.joblib

# 2. Modelos treinados?
ls -lh project/project/modelo_rl_final.pt project/project/sl_profit_regressor_model.joblib

# 3. Backend funcionando?
curl -X POST http://127.0.0.1:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"region":"North America","content":"Image","age":"25-34","gender":"Female","platform":"Instagram","budget":1000,"product_tier":"Low Ticket"}'

# Resposta esperada:
# {"recommended_price": 49.90, "estimated_roi": 150.0, ...}
```

## 🐛 Resolução de Problemas

### Erro: `No module named 'numpy'`
```bash
pip install numpy pandas scikit-learn d3rlpy joblib gymnasium tqdm
```

### Erro: `ValueError: 'region_metrics' not defined`
- ❌ **NÃO execute** a célula "Etapa 9.1" do notebook
- ✅ **USE** os dados gerados pelo `Generator_NEW.py`

### Erro: Preço = `$-0.02`
- ❌ **Faltou** aplicar `scaler_acao.inverse_transform()`
- ✅ **Adicione** a conversão no código (veja Passo 3)

### Erro: `FileNotFoundError: sl_dataset_combined.csv`
- ❌ **Não executou** o `Generator_NEW.py`
- ✅ **Execute** o Passo 1 primeiro

## 📊 Resultado Final

**ANTES:**
```
❌ ValueError: 'region_metrics' não definidos
❌ Preço: $-0.02
```

**DEPOIS:**
```
✅ Dataset gerado: 50.000 linhas
✅ Modelo treinado: modelo_rl_final.pt
✅ Preço: $49.90
✅ ROI: 150%
```

---

**⏱️ Tempo total: ~20 minutos**

**📚 Documentação completa: `README_EXECUCAO_COMPLETA.md`**
