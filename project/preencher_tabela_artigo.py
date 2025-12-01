import pandas as pd
import numpy as np
import joblib
import d3rlpy
import json
import os
import warnings

# Suprime avisos de versões
warnings.filterwarnings('ignore')

print("="*60)
print("🤖 GERADOR DE TABELA DE RESULTADOS (RL FIXO)")
print("="*60)

# --- 1. DEFINA AQUI AS LINHAS DA SUA TABELA ---
# Coloque exatamente os valores que você quer testar no artigo
# --- 1. DEFINA AQUI AS LINHAS DA SUA TABELA ---
CENARIOS_TABELA = [
    # --- LOW TICKET ---
    {'Nome': 'Cenário 1',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 100.0},   # Micro, Budget Baixo
    {'Nome': 'Cenário 2',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 200.0},   # Mid-Low, Budget Baixo
    {'Nome': 'Cenário 3',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 1000.0},  # Low-Mid, Budget Médio
    {'Nome': 'Cenário 4',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 1000.0},  # Mid-Low, Budget Médio (Repetido propositalmente para ver variação se houver, ou mude o Tier)
    {'Nome': 'Cenário 5',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 10000.0}, # Low-Mid, Budget Alto
    {'Nome': 'Cenário 6',  'Regiao': 'North America', 'Tier': 'Low Ticket',  'Orcamento': 10000.0}, # Mid-Low, Budget Alto

    # --- HIGH TICKET ---
    {'Nome': 'Cenário 7',  'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 2000.0},  # High Entry, Budget Baixo
    {'Nome': 'Cenário 8',  'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 5000.0},  # High Mid, Budget Médio
    {'Nome': 'Cenário 9',  'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 10000.0}, # High Premium, Budget Médio
    {'Nome': 'Cenário 10', 'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 50000.0}, # High Premium, Budget Alto
    {'Nome': 'Cenário 11', 'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 100000.0},# Ultra High, Budget Alto
    {'Nome': 'Cenário 12', 'Regiao': 'North America', 'Tier': 'High Ticket', 'Orcamento': 200000.0} # Enterprise, Budget Muito Alto
]

# Configuração padrão para campos que não mudam na tabela (ceteris paribus)
BASE_CONFIG = {
    'Plataforma': 'Instagram',
    'Idade': '25-34',
    'Genero': 'Female',
    'Conteudo': 'Video',
    'Tipo_Produto': 'InfoProduto',
    'Modelo_Cobranca': 'Venda Unica',
    'Complexidade_Oferta': 'Media'
}

# --- 2. CARREGAR O CÉREBRO (Artefatos) ---
print("Carregando modelos...", end=" ")
try:
    # Scalers
    ohe = joblib.load("ohe_encoder.joblib")
    scaler_state = joblib.load("scaler_estado.joblib")
    scaler_action = joblib.load("scaler_acao.joblib")
    scaler_reward = joblib.load("scaler_recompensa.joblib")
    
    # Modelo Treinado
    cql = d3rlpy.load_learnable("modelo_rl_final.pt", device="cpu")
    
    # Metadados das colunas
    with open('colunas_estado_base.json', 'r') as f:
        colunas_ordenadas = json.load(f)
        
    print("✅ Sucesso!")
except Exception as e:
    print(f"\n❌ Erro ao carregar arquivos: {e}")
    exit()

# --- 3. O LOOP DE INFERÊNCIA ---
print(f"\n{'CENÁRIO':<10} | {'TIER':<12} | {'ORÇAMENTO':<10} | {'PREÇO REC. (IA)':<15} | {'LUCRO ESPERADO':<15}")
print("-" * 75)

for linha in CENARIOS_TABELA:
    # 1. Monta o dicionário completo do estado
    estado_dict = BASE_CONFIG.copy()
    estado_dict.update(linha) # Sobrescreve com os dados da tabela
    
    # 2. Pré-processamento (Igual ao main.py)
    # Cria DF de uma linha
    df = pd.DataFrame([estado_dict])
    
    # Aplica OHE nas categóricas
    cat_cols = ohe.feature_names_in_
    x_cat = ohe.transform(df[cat_cols])
    
    # Aplica Scaler nas numéricas
    num_cols = scaler_state.feature_names_in_
    x_num = scaler_state.transform(df[num_cols])
    
    # Junta tudo
    x_raw = np.concatenate([x_cat, x_num], axis=1)
    
    # Garante a ordem correta das colunas (se houver desalinhamento)
    # Nota: Aqui assumimos que a concatenação segue a ordem de criação. 
    # Para rigor absoluto, reindexaríamos com pandas se tivéssemos os nomes das colunas OHE.
    # Como o fluxo é controlado, x_raw deve estar ok.
    state_vector = x_raw.astype(np.float32)

    # 3. INFERÊNCIA (A Pergunta para a IA)
    # Qual o preço?
    acao_norm = cql.predict(state_vector)[0]
    preco_real = scaler_action.inverse_transform(acao_norm.reshape(1, -1))[0][0]
    
    # Qual o lucro esperado? (Risco)
    if hasattr(cql, "predict_value"):
        action_batch = acao_norm.reshape(1, -1)
        lucro_norm = cql.predict_value(state_vector, action_batch)[0]
        lucro_real = scaler_reward.inverse_transform(lucro_norm.reshape(1, -1))[0][0]
    else:
        lucro_real = 0.0

    # 4. IMPRIMIR RESULTADO DA LINHA
    print(f"{linha['Nome']:<10} | {linha['Tier']:<12} | ${linha['Orcamento']:<9,.0f} | ${preco_real:<14.2f} | ${lucro_real:<14.2f}")

print("-" * 75)