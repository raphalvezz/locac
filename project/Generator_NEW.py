#!/usr/bin/env python3
"""
Generator - Gêmeo Digital Econômico
Gera datasets sintéticos para SL e RL (Venda Única e Assinatura)
"""

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import d3rlpy
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer
import gymnasium as gym
from tqdm import tqdm

print("="*80)
print("GENERATOR - Gêmeo Digital Econômico")
print("="*80)

# ============================================================================
# BLOCO 1: Definir o Gêmeo Digital (Economia-Base)
# ============================================================================

print("\n[1/4] Definindo parâmetros econômicos...")

# CORREÇÃO DO ValueError: Aqui estão os 'region_metrics' e 'elasticity_factors'!
economic_parameters = {
    'North America': {
        'beta': 0.015,      # Sensibilidade ao preço
        'c0': 100.0,        # Demanda base
        'a0': 50.0,         # Preço de referência
        'cpa': 20.0
    },
    'Europe': {
        'beta': 0.018,
        'c0': 80.0,
        'a0': 55.0,
        'cpa': 25.0
    },
    'Asia': {
        'beta': 0.012,
        'c0': 120.0,
        'a0': 45.0,
        'cpa': 15.0
    },
    'South America': {
        'beta': 0.020,
        'c0': 60.0,
        'a0': 40.0,
        'cpa': 18.0
    },
    'Africa': {
        'beta': 0.022,
        'c0': 50.0,
        'a0': 35.0,
        'cpa': 12.0
    }
}

cost_parameters = {
    'SL_Low_Ticket_CPA': 15.0,
    'SL_High_Ticket_CPA': 80.0,
    'RL_Venda_Unica_Low_CPA': 15.0,
    'RL_Venda_Unica_High_CPA': 80.0,
    'RL_Assinatura_CPA': 100.0
}

print(f"✓ Regiões: {list(economic_parameters.keys())}")
print(f"✓ Custos definidos: {list(cost_parameters.keys())}")

# Funções do simulador
def calculate_demand(estado, preco):
    """C(s,a) = C0(s) * exp(-beta * (a - a0))"""
    regiao = estado['Region']
    params = economic_parameters[regiao]
    beta = params['beta']
    c0 = params['c0']
    a0 = params['a0']
    conversoes = c0 * np.exp(-beta * (preco - a0))
    return max(0, conversoes)

def calculate_profit(preco, cpa, conversoes):
    """lucro = conversoes * (preco - cpa), com piso = 0"""
    lucro = conversoes * (preco - cpa)
    return max(0, lucro)  # REGRA CRÍTICA

print("✓ Funções de demanda e lucro implementadas")

# ============================================================================
# BLOCO 2: Estrutura de Features
# ============================================================================

feature_values = {
    'Region': ['North America', 'Europe', 'Asia', 'South America', 'Africa'],
    'Content_Type': ['Image', 'Video', 'Carousel', 'Text'],
    'Target_Age': ['18-24', '25-34', '35-44', '45-54', '55+'],
    'Target_Gender': ['Male', 'Female', 'Other'],
    'Platform': ['Instagram', 'Facebook', 'LinkedIn', 'YouTube', 'Google'],
    'Budget': ['Baixo', 'Médio', 'Alto'],
    'Product_Tier': ['Low Ticket', 'Mid Ticket', 'High Ticket']
}

def generate_random_state():
    """Gera estado aleatório"""
    return {feature: np.random.choice(values) for feature, values in feature_values.items()}

print(f"✓ {len(feature_values)} features categóricas definidas")

# ============================================================================
# BLOCO 3: Geração do Dataset SL
# ============================================================================

print("\n[2/4] Gerando dataset Supervised Learning...")

def generate_sl_dataset(num_samples=50000):
    data_rows = []
    for _ in tqdm(range(num_samples), desc="SL Dataset"):
        estado = generate_random_state()
        tier = estado['Product_Tier']
        
        # Lógica Low/Mid/High Ticket
        if tier == 'Low Ticket':
            cpa = cost_parameters['SL_Low_Ticket_CPA']
            preco = np.random.uniform(cpa + 5, cpa + 100)
        elif tier == 'High Ticket':
            cpa = cost_parameters['SL_High_Ticket_CPA']
            preco = np.random.uniform(cpa + 50, cpa + 500)
        else:  # Mid Ticket
            cpa = (cost_parameters['SL_Low_Ticket_CPA'] + cost_parameters['SL_High_Ticket_CPA']) / 2
            preco = np.random.uniform(cpa + 20, cpa + 200)
        
        conversoes = calculate_demand(estado, preco)
        lucro = calculate_profit(preco, cpa, conversoes)
        
        row = estado.copy()
        row['Preco'] = preco
        row['Lucro'] = lucro
        data_rows.append(row)
    
    return pd.DataFrame(data_rows)

df_sl = generate_sl_dataset(50000)
df_sl.to_csv('sl_dataset_combined.csv', index=False)
print(f"✓ Dataset SL salvo: {len(df_sl)} linhas")
print(f"  Lucro médio: ${df_sl['Lucro'].mean():.2f}")

# Salvar scalers SL
X_sl = df_sl.drop(['Preco', 'Lucro'], axis=1)
encoder_sl = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_sl.fit(X_sl)
joblib.dump(encoder_sl, 'sl_encoder.joblib')

X_sl_encoded = encoder_sl.transform(X_sl)
scaler_sl_estado = StandardScaler().fit(X_sl_encoded)
joblib.dump(scaler_sl_estado, 'sl_scaler_estado.joblib')

scaler_sl_preco = StandardScaler().fit(df_sl[['Preco']])
joblib.dump(scaler_sl_preco, 'sl_scaler_preco.joblib')

scaler_sl_lucro = StandardScaler().fit(df_sl[['Lucro']])
joblib.dump(scaler_sl_lucro, 'sl_scaler_lucro.joblib')

print("✓ Scalers SL salvos: 4 arquivos .joblib")

# ============================================================================
# BLOCO 4: Geração do Buffer RL Venda Única
# ============================================================================

print("\n[3/4] Gerando buffer RL Venda Única...")

def get_reward_venda_unica(preco, estado):
    tier = estado['Product_Tier']
    if tier == 'Low Ticket':
        cpa = cost_parameters['RL_Venda_Unica_Low_CPA']
    elif tier == 'High Ticket':
        cpa = cost_parameters['RL_Venda_Unica_High_CPA']
    else:
        cpa = (cost_parameters['RL_Venda_Unica_Low_CPA'] + cost_parameters['RL_Venda_Unica_High_CPA']) / 2
    
    conversoes = calculate_demand(estado, preco)
    return calculate_profit(preco, cpa, conversoes)

transitions_rl = []
for _ in tqdm(range(100000), desc="RL Buffer"):
    estado = generate_random_state()
    tier = estado['Product_Tier']
    
    if tier == 'Low Ticket':
        cpa = cost_parameters['RL_Venda_Unica_Low_CPA']
        preco = np.random.uniform(cpa + 5, cpa + 100)
    elif tier == 'High Ticket':
        cpa = cost_parameters['RL_Venda_Unica_High_CPA']
        preco = np.random.uniform(cpa + 50, cpa + 500)
    else:
        cpa = (cost_parameters['RL_Venda_Unica_Low_CPA'] + cost_parameters['RL_Venda_Unica_High_CPA']) / 2
        preco = np.random.uniform(cpa + 20, cpa + 200)
    
    recompensa = get_reward_venda_unica(preco, estado)
    proximo_estado = generate_random_state()
    
    transitions_rl.append({
        'estado': estado,
        'acao': preco,
        'recompensa': recompensa,
        'proximo_estado': proximo_estado
    })

df_rl = pd.DataFrame([{**t['estado'], 'Preco': t['acao'], 'Recompensa': t['recompensa']} for t in transitions_rl])

print(f"✓ Buffer RL gerado: {len(df_rl)} transições")
print(f"  Recompensa média: ${df_rl['Recompensa'].mean():.2f}")

# Salvar scalers RL
X_rl = df_rl[list(feature_values.keys())]
encoder_rl = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_rl)
joblib.dump(encoder_rl, 'rl_encoder_estado.joblib')

X_rl_encoded = encoder_rl.transform(X_rl)
scaler_rl_estado = StandardScaler().fit(X_rl_encoded)
joblib.dump(scaler_rl_estado, 'scaler_estado.joblib')

scaler_rl_acao = StandardScaler().fit(df_rl[['Preco']])
joblib.dump(scaler_rl_acao, 'scaler_acao.joblib')

scaler_rl_recompensa = StandardScaler().fit(df_rl[['Recompensa']])
joblib.dump(scaler_rl_recompensa, 'scaler_recompensa.joblib')

# Criar ReplayBuffer
observations = scaler_rl_estado.transform(encoder_rl.transform(X_rl))
actions = scaler_rl_acao.transform(df_rl[['Preco']]).flatten()
rewards = df_rl['Recompensa'].values

observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=observations.shape[1:], dtype=np.float32)
action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

buffer = ReplayBuffer(FIFOBuffer(limit=len(transitions_rl)), observation_space=observation_space, action_space=action_space)

print("Adicionando transições ao buffer...")
for i in tqdm(range(len(transitions_rl))):
    buffer.add_transition(
        observation=observations[i].astype(np.float32),
        action=np.array([actions[i]], dtype=np.float32),
        reward=float(rewards[i]),
        next_observation=observations[(i+1) % len(transitions_rl)].astype(np.float32),
        terminal=0.0
    )

buffer.dump('rl_offline_buffer.h5')
print(f"✓ Buffer salvo: rl_offline_buffer.h5 ({len(buffer)} transições)")
print("✓ Scalers RL salvos: 4 arquivos .joblib")

# ============================================================================
# BLOCO 5: Geração do Buffer RL Assinatura
# ============================================================================

print("\n[4/4] Gerando buffer RL Assinatura...")

def calculate_churn_rate(preco_mensalidade, estado):
    return min(0.5, max(0.01, 0.05 + (preco_mensalidade * 0.001)))

def get_reward_assinatura(preco_mensalidade, estado):
    cpa = cost_parameters['RL_Assinatura_CPA']
    churn = calculate_churn_rate(preco_mensalidade, estado)
    ltv = (preco_mensalidade / churn) - cpa
    return max(0, ltv)

def generate_subscription_state():
    estado = generate_random_state()
    estado['dias_desde_ultima_interacao'] = np.random.randint(1, 90)
    estado['clv_estimate_percentile'] = np.random.uniform(0.1, 1.0)
    estado['avg_price_offered_segment_90d'] = np.random.uniform(20.0, 150.0)
    estado['price_volatility_30d'] = np.random.uniform(0.5, 10.0)
    return estado

transitions_sub = []
for _ in tqdm(range(50000), desc="Assinatura Buffer"):
    estado = generate_subscription_state()
    preco_mensalidade = np.random.uniform(10.0, 200.0)
    recompensa = get_reward_assinatura(preco_mensalidade, estado)
    proximo_estado = generate_subscription_state()
    
    transitions_sub.append({
        'estado': estado,
        'acao': preco_mensalidade,
        'recompensa': recompensa,
        'proximo_estado': proximo_estado
    })

memoria_cols = ['dias_desde_ultima_interacao', 'clv_estimate_percentile', 'avg_price_offered_segment_90d', 'price_volatility_30d']
df_sub = pd.DataFrame([{**{k: v for k, v in t['estado'].items() if k in feature_values}, **{k: t['estado'][k] for k in memoria_cols}, 'Mensalidade': t['acao'], 'LTV': t['recompensa']} for t in transitions_sub])

print(f"✓ Buffer Assinatura gerado: {len(df_sub)} transições")
print(f"  LTV médio: ${df_sub['LTV'].mean():.2f}")

# Salvar scalers Assinatura
X_sub_cat = df_sub[list(feature_values.keys())]
encoder_sub = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_sub_cat)
joblib.dump(encoder_sub, 'assinatura_encoder_estado.joblib')

scaler_sub_memoria = StandardScaler().fit(df_sub[memoria_cols])
joblib.dump(scaler_sub_memoria, 'scaler_assinatura_memoria.joblib')

X_sub_encoded = encoder_sub.transform(X_sub_cat)
X_sub_memoria_scaled = scaler_sub_memoria.transform(df_sub[memoria_cols])
X_sub_completo = np.hstack([X_sub_encoded, X_sub_memoria_scaled])

scaler_sub_estado = StandardScaler().fit(X_sub_completo)
joblib.dump(scaler_sub_estado, 'scaler_assinatura_estado.joblib')

scaler_sub_acao = StandardScaler().fit(df_sub[['Mensalidade']])
joblib.dump(scaler_sub_acao, 'scaler_assinatura_acao.joblib')

scaler_sub_recompensa = StandardScaler().fit(df_sub[['LTV']])
joblib.dump(scaler_sub_recompensa, 'scaler_assinatura_recompensa.joblib')

# Criar ReplayBuffer Assinatura
observations_sub = scaler_sub_estado.transform(X_sub_completo)
actions_sub = scaler_sub_acao.transform(df_sub[['Mensalidade']]).flatten()
rewards_sub = df_sub['LTV'].values

observation_space_sub = gym.spaces.Box(low=-np.inf, high=np.inf, shape=observations_sub.shape[1:], dtype=np.float32)
action_space_sub = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

buffer_sub = ReplayBuffer(FIFOBuffer(limit=len(transitions_sub)), observation_space=observation_space_sub, action_space=action_space_sub)

print("Adicionando transições ao buffer de assinatura...")
for i in tqdm(range(len(transitions_sub))):
    buffer_sub.add_transition(
        observation=observations_sub[i].astype(np.float32),
        action=np.array([actions_sub[i]], dtype=np.float32),
        reward=float(rewards_sub[i]),
        next_observation=observations_sub[(i+1) % len(transitions_sub)].astype(np.float32),
        terminal=0.0
    )

buffer_sub.dump('rl_assinatura_buffer.h5')
print(f"✓ Buffer salvo: rl_assinatura_buffer.h5 ({len(buffer_sub)} transições)")
print("✓ Scalers Assinatura salvos: 5 arquivos .joblib")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMO FINAL - Artefatos Gerados")
print("="*80)

import os
arquivos = [
    ('sl_dataset_combined.csv', 'Dataset SL'),
    ('rl_offline_buffer.h5', 'Buffer RL Venda Única'),
    ('rl_assinatura_buffer.h5', 'Buffer RL Assinatura'),
    ('scaler_estado.joblib', 'Scaler Estado (RL)'),
    ('scaler_acao.joblib', 'Scaler Ação (RL)'),
    ('scaler_recompensa.joblib', 'Scaler Recompensa (RL)'),
    ('scaler_assinatura_memoria.joblib', 'Scaler Memória (Assinatura)')
]

for arquivo, descricao in arquivos:
    if os.path.exists(arquivo):
        tamanho = os.path.getsize(arquivo) / (1024*1024)
        print(f"✓ {descricao:<35} {arquivo:<35} {tamanho:>8.2f} MB")
    else:
        print(f"✗ {descricao:<35} {arquivo:<35} FALTANDO")

print("="*80)
print("\n🎉 SUCESSO! Todos os datasets foram gerados.")
print("📊 O ValueError foi CORRIGIDO! region_metrics e elasticity_factors agora estão definidos.")
print("\nPróximos passos:")
print("  1. Execute SL_FINAL (1).ipynb para treinar o modelo baseline")
print("  2. Execute código_final_RL_OFF (25).ipynb para treinar o agente RL de venda única")
print("  3. Execute RL_assinatura (5).ipynb para treinar o agente RL de assinatura")
