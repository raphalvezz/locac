#!/usr/bin/env python3
"""
Generator - Gêmeo Digital Econômico (v3 - Dinâmico)
Gera datasets sintéticos para SL e RL (Venda Única e Assinatura)
Implementa lógica dinâmica de preço e escala definida pelo usuário.
"""

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import d3rlpy
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode
import gymnasium as gym
from tqdm import tqdm
import os
import shutil

print("="*80)
print("GENERATOR (v3 - Dinâmico) - Gêmeo Digital Econômico")
print("="*80)

# ============================================================================
# CONFIGURAÇÃO DO USUÁRIO (Faixas de Preço)
# ============================================================================
# O usuário define o que é "Low" e "High" para o negócio dele.
# O sistema ajustará automaticamente a demanda e custos baseado nestes valores.
USER_PRICE_RANGES = {
    'Low Ticket':  {'min': 10.0,  'max': 97.0},   # Ex: E-books, cursos básicos
    'High Ticket': {'min': 497.0, 'max': 5000.0}  # Ex: Mentorias, Consultorias
}
# NOVO: Faixa de Orçamento (Mensal)
# Define o mínimo e máximo que o simulador vai considerar para os orçamentos
USER_BUDGET_RANGE = {'min': 500.0, 'max': 20000.0}
# Define a agressividade do mercado (CPA como % do Preço)
# Ex: Em média, gasta-se 20% a 45% do preço para fazer uma venda.
CPA_RATIO_RANGE = (0.20, 0.45)

# ============================================================================
# BLOCO 1: Definir o Gêmeo Digital (Parâmetros Base)
# ============================================================================

print("\n[1/5] Definindo parâmetros econômicos dinâmicos...")

# Parâmetros Econômicos Base (Apenas sensibilidade regional e demanda base relativa)
# 'a0' e 'cpa' fixos foram REMOVIDOS pois agora são calculados dinamicamente.
economic_parameters = {
    'North America': {'beta': 0.015, 'c0': 100.0},
    'Europe':        {'beta': 0.018, 'c0': 80.0},
    'Asia':          {'beta': 0.012, 'c0': 120.0},
    'South America': {'beta': 0.020, 'c0': 70.0},
    'Africa':        {'beta': 0.022, 'c0': 60.0}
}

# Features Categóricas e seus valores possíveis
feature_values = {
    'Regiao': ['North America', 'Europe', 'Asia', 'South America', 'Africa'],
    'Plataforma': ['Instagram', 'Facebook', 'LinkedIn', 'Twitter', 'Pinterest'],
    'Tier': ['Low Ticket', 'High Ticket'],
    'Idade': ['18-24', '25-34', '35-44', '45-54', '55+'],
    'Genero': ['Female', 'Male', 'Other'],
    'Conteudo': ['Image', 'Video', 'Text', 'Carousel'],
    'Tipo_Produto': ['SaaS', 'InfoProduto', 'Serviço'],
    'Modelo_Cobranca': ['Assinatura', 'Venda Unica'],
    'Complexidade_Oferta': ['Baixa', 'Media', 'Alta']
}

categorical_features = list(feature_values.keys())
numeric_features_base = ['Orcamento']

# Features numéricas (estado de memória, apenas para Assinatura)
numeric_features_memoria = [
    'dias_desde_ultima_interacao',
    'clv_estimate_percentile',
    'avg_price_offered_segment_90d',
    'price_volatility_30d'
]

# --- Funções Dinâmicas do Gêmeo Digital ---

def generate_dynamic_price_cpa(tier):
    """
    Gera um preço e um CPA coerentes com o Tier selecionado e as configurações do usuário.
    """
    # Pega a faixa definida pelo usuário
    price_range = USER_PRICE_RANGES.get(tier, {'min': 10, 'max': 100})
    
    # Escolhe um preço aleatório dentro da faixa
    price = np.random.uniform(price_range['min'], price_range['max'])
    
    # O CPA (Custo) modela a realidade: produtos caros aceitam CPAs maiores.
    cpa_ratio = np.random.uniform(CPA_RATIO_RANGE[0], CPA_RATIO_RANGE[1])
    cpa = price * cpa_ratio
    
    return price, cpa

def get_demand_modifiers(estado):
    """Calcula modificadores de demanda baseados no produto."""
    c0_factor = 1.0
    beta_factor = 1.0

    # Influência do Tipo de Produto
    if estado['Tipo_Produto'] == 'SaaS':
        c0_factor *= 1.2
        beta_factor *= 0.9 
    elif estado['Tipo_Produto'] == 'InfoProduto':
        c0_factor *= 1.0
        beta_factor *= 1.1 
    elif estado['Tipo_Produto'] == 'Serviço':
        c0_factor *= 0.8
        beta_factor *= 1.0

    # Influência da Complexidade
    if estado['Complexidade_Oferta'] == 'Alta':
        c0_factor *= 0.7 
        beta_factor *= 0.8 
    elif estado['Complexidade_Oferta'] == 'Baixa':
        c0_factor *= 1.3 
        beta_factor *= 1.2 

    return c0_factor, beta_factor

def calculate_demand_dynamic(estado, preco):
    """
    Calcula a demanda ajustada dinamicamente à escala de preço do usuário.
    """
    tier = estado['Tier']
    
    # O preço de referência (a0) agora é a MÉDIA da faixa do usuário.
    # Isso garante que a curva funcione tanto para $50 quanto para $5000.
    range_info = USER_PRICE_RANGES.get(tier, {'min': 10, 'max': 100})
    a0_dynamic = (range_info['min'] + range_info['max']) / 2
    
    # Parâmetros base regionais
    regiao = estado['Regiao']
    params = economic_parameters.get(regiao, economic_parameters['North America'])
    
    # Modificadores de Produto
    c0_prod_factor, beta_prod_factor = get_demand_modifiers(estado)

    # Demanda Base (c0) ajustada pela Escala (Elasticidade de Escala)
    # Se o produto custa $5000, a demanda natural numérica é menor que um de $50.
    # Normaliza baseado em um produto "padrão" de $50.
    scale_factor = 50.0 / a0_dynamic 
    c0_final = params['c0'] * c0_prod_factor * np.sqrt(scale_factor)
    
    beta_final = params['beta'] * beta_prod_factor
    
    # Influência do Orçamento (Logarítmica)
    orcamento_factor = np.log1p(estado['Orcamento']) / np.log1p(5000)
    c0_final *= (0.5 + orcamento_factor)

    # Fórmula de demanda ajustada relativa ao a0 dinâmico
    # (preco - a0) é normalizado pelo próprio a0 para manter a exp() estável
    conversoes = c0_final * np.exp(-beta_final * ((preco - a0_dynamic) / (a0_dynamic * 0.1) ))
    
    return max(0, conversoes)

def calculate_churn_rate_dynamic(estado, preco_mensalidade):
    """Calcula churn adaptado à escala de preço."""
    tier = estado['Tier']
    range_info = USER_PRICE_RANGES.get(tier, {'min': 10, 'max': 100})
    a0_dynamic = (range_info['min'] + range_info['max']) / 2

    churn_base = 0.05
    
    # O impacto do preço no churn agora é relativo à média da faixa
    # Se o preço é alto para a categoria, churn sobe.
    price_factor = (preco_mensalidade / a0_dynamic) * 0.02
    
    # Memória e Ancoragem
    anchor_price = estado.get('avg_price_offered_segment_90d', a0_dynamic)
    anchor_diff = (preco_mensalidade - anchor_price) / anchor_price
    anchor_factor = max(0, anchor_diff) * 0.1
    
    dias = estado.get('dias_desde_ultima_interacao', 30)
    dias_factor = (dias / 90) * 0.05
    
    churn_final = churn_base + price_factor + anchor_factor + dias_factor
    return float(min(max(churn_final, 0.01), 0.8))

# --- Estrutura de Transição (para d3rlpy) ---
class Transition:
    def __init__(self, observation, action, reward, terminal):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal

print("✓ Lógica Dinâmica configurada.")
print(f"✓ Faixas de Preço do Usuário: {USER_PRICE_RANGES}")


# ============================================================================
# BLOCO 2: Gerar Dataset Supervised Learning (SL)
# ============================================================================

print("\n[2/5] Gerando dataset Supervised Learning (Dinâmico)...")

def generate_sl_dataset(num_samples=50000):
    data = []
    
    for _ in tqdm(range(num_samples)):
        # 1. Gera estado aleatório
        estado = {feat: np.random.choice(feature_values[feat]) for feat in categorical_features}
        estado['Orcamento'] = np.random.uniform(USER_BUDGET_RANGE['min'], USER_BUDGET_RANGE['max'])
        
        # 2. Gera Preço e CPA dinâmicos baseados no Tier do estado
        preco_amostra, cpa_dinamico = generate_dynamic_price_cpa(estado['Tier'])
        
        # 3. Calcula Lucro (Venda Unica ou LTV)
        if estado['Modelo_Cobranca'] == 'Venda Unica':
            conversoes = calculate_demand_dynamic(estado, preco_amostra)
            lucro_real = conversoes * (preco_amostra - cpa_dinamico)
        else: # Assinatura
            # Mock de memória para SL
            estado['avg_price_offered_segment_90d'] = preco_amostra
            churn = calculate_churn_rate_dynamic(estado, preco_amostra)
            if churn > 0:
                ltv = (preco_amostra / churn) - cpa_dinamico
                lucro_real = ltv
            else:
                lucro_real = 0

        # Regra de Piso de Lucro
        lucro_real = max(0, lucro_real)

        # 4. Adiciona ao dataset
        row = estado.copy()
        row['Preco_Amostra'] = preco_amostra
        row['Lucro_Real'] = lucro_real
        data.append(row)
        
    df = pd.DataFrame(data)
    df = df.drop(columns=numeric_features_memoria, errors='ignore')
    return df

df_sl = generate_sl_dataset()
df_sl.to_csv('sl_dataset_combined.csv', index=False)

print(f"✓ Dataset SL salvo: {len(df_sl)} linhas")
print(f"  Lucro médio (SL): ${df_sl['Lucro_Real'].mean():.2f}")

# Salvar Pré-processadores SL
ohe_sl = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df_sl[categorical_features])
scaler_sl = StandardScaler().fit(df_sl[numeric_features_base])
scaler_preco_sl = StandardScaler().fit(df_sl[['Preco_Amostra']])
scaler_lucro_sl = StandardScaler().fit(df_sl[['Lucro_Real']])

joblib.dump(ohe_sl, 'sl_encoder.joblib')
joblib.dump(scaler_sl, 'sl_scaler_estado.joblib')
joblib.dump(scaler_preco_sl, 'sl_scaler_preco.joblib')
joblib.dump(scaler_lucro_sl, 'sl_scaler_lucro.joblib')
print("✓ Scalers SL salvos.")

# ============================================================================
# BLOCO 3: Gerar Buffer RL Venda Única (Dinâmico)
# ============================================================================

print("\n[3/5] Gerando buffer RL Venda Única (Dinâmico)...")

def generate_rl_offline_buffer(num_samples=100000):
    transitions_rl = []
    
    for _ in tqdm(range(num_samples)):
        # 1. Gera estado (apenas Venda Unica)
        estado = {feat: np.random.choice(feature_values[feat]) for feat in categorical_features}
        estado['Modelo_Cobranca'] = 'Venda Unica' 
        estado['Orcamento'] = np.random.uniform(USER_BUDGET_RANGE['min'], USER_BUDGET_RANGE['max'])

        # 2. Gera Preço e CPA Dinâmicos
        preco_amostra, cpa_dinamico = generate_dynamic_price_cpa(estado['Tier'])
        
        # 3. Calcula Demanda e Lucro
        conversoes = calculate_demand_dynamic(estado, preco_amostra)
        recompensa = max(0, conversoes * (preco_amostra - cpa_dinamico))
        
        # 4. Cria transição
        transitions_rl.append(Transition(
            observation=estado,
            action=preco_amostra,
            reward=recompensa,
            terminal=0.0
        ))
        
    return transitions_rl

transitions_rl = generate_rl_offline_buffer()

# --- Pré-processar e Salvar Buffer RL Venda Única ---
ohe_rl = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(
    pd.DataFrame([t.observation for t in transitions_rl])[categorical_features]
)
scaler_estado_rl = StandardScaler().fit(
    pd.DataFrame([t.observation for t in transitions_rl])[numeric_features_base]
)
scaler_acao_rl = StandardScaler().fit(
    np.array([t.action for t in transitions_rl]).reshape(-1, 1)
)
scaler_recompensa_rl = StandardScaler().fit(
    np.array([t.reward for t in transitions_rl]).reshape(-1, 1)
)

# Salva colunas
colunas_estado_base = list(ohe_rl.get_feature_names_out()) + numeric_features_base
with open('colunas_estado_base.json', 'w') as f:
    json.dump(colunas_estado_base, f)

# Processamento dos dados para o Buffer
df_obs_rl = pd.DataFrame([t.observation for t in transitions_rl])
processed_observations = np.concatenate([
    ohe_rl.transform(df_obs_rl[categorical_features]),
    scaler_estado_rl.transform(df_obs_rl[numeric_features_base])
], axis=1)

processed_actions = scaler_acao_rl.transform(np.array([t.action for t in transitions_rl]).reshape(-1, 1))
processed_rewards = scaler_recompensa_rl.transform(np.array([t.reward for t in transitions_rl]).reshape(-1, 1)).flatten()

# Criar Episode e Buffer (Correção d3rlpy v2)
# (CÓDIGO NOVO / CORRIGIDO)
print("Convertendo para Episódio RL...")
# Correção: Removemos 'terminals=' e passamos 'False' (booleano) na 4ª posição.
# A ordem é: (observations, actions, rewards, terminated)
episode_rl = Episode(
    processed_observations.astype(np.float32),
    processed_actions.astype(np.float32),
    processed_rewards.reshape(-1, 1).astype(np.float32),
    False # terminated=False (O episódio nunca acaba neste dataset sintético)
)
buffer = ReplayBuffer(FIFOBuffer(limit=len(transitions_rl)), episodes=[episode_rl])
with open('rl_offline_buffer.h5', 'w+b') as f:
    buffer.dump(f)

# Salva scalers
joblib.dump(ohe_rl, 'ohe_encoder.joblib')
joblib.dump(scaler_estado_rl, 'scaler_estado.joblib')
joblib.dump(scaler_acao_rl, 'scaler_acao.joblib')
joblib.dump(scaler_recompensa_rl, 'scaler_recompensa.joblib')

print(f"✓ Buffer RL Venda salvo: {len(transitions_rl)} transições.")

# ============================================================================
# BLOCO 4: Gerar Buffer RL Assinatura (Dinâmico)
# ============================================================================

print("\n[4/5] Gerando buffer RL Assinatura (Dinâmico)...")

def generate_rl_assinatura_buffer(num_samples=50000):
    transitions_sub = []
    
    for _ in tqdm(range(num_samples)):
        # 1. Gera estado
        estado = {feat: np.random.choice(feature_values[feat]) for feat in categorical_features}
        estado['Modelo_Cobranca'] = 'Assinatura'
        estado['Orcamento'] = np.random.uniform(USER_BUDGET_RANGE['min'], USER_BUDGET_RANGE['max'])
        
        # Features de memória
        estado['dias_desde_ultima_interacao'] = np.random.uniform(1, 90)
        estado['clv_estimate_percentile'] = np.random.uniform(0.1, 1.0)
        
        # Preço de referência dinâmico baseado no Tier
        tier = estado['Tier']
        range_info = USER_PRICE_RANGES.get(tier, {'min': 10, 'max': 100})
        a0_dynamic = (range_info['min'] + range_info['max']) / 2
        estado['avg_price_offered_segment_90d'] = np.random.uniform(a0_dynamic * 0.8, a0_dynamic * 1.2)
        estado['price_volatility_30d'] = np.random.uniform(0.5, 10.0)

        # 2. Gera Ação (Mensalidade) e CPA Dinâmicos
        preco_amostra, cpa_dinamico = generate_dynamic_price_cpa(tier)
        
        # 3. Calcula LTV (Recompensa)
        churn = calculate_churn_rate_dynamic(estado, preco_amostra)
        if churn > 0:
            recompensa = max(0, (preco_amostra / churn) - cpa_dinamico)
        else:
            recompensa = 0.0
        
        # 4. Cria transição
        transitions_sub.append(Transition(
            observation=estado,
            action=preco_amostra,
            reward=recompensa,
            terminal=0.0
        ))
        
    return transitions_sub

transitions_sub = generate_rl_assinatura_buffer()

# --- Pré-processar e Salvar Buffer Assinatura ---
df_obs_sub = pd.DataFrame([t.observation for t in transitions_sub])

scaler_memoria_sub = StandardScaler().fit(df_obs_sub[numeric_features_memoria])
scaler_acao_sub = StandardScaler().fit(np.array([t.action for t in transitions_sub]).reshape(-1, 1))
scaler_recompensa_sub = StandardScaler().fit(np.array([t.reward for t in transitions_sub]).reshape(-1, 1))

colunas_estado_assinatura = list(ohe_rl.get_feature_names_out()) + numeric_features_base + numeric_features_memoria
with open('colunas_estado_assinatura.json', 'w') as f:
    json.dump(colunas_estado_assinatura, f)

# Processamento
processed_observations_sub = np.concatenate([
    ohe_rl.transform(df_obs_sub[categorical_features]),
    scaler_estado_rl.transform(df_obs_sub[numeric_features_base]),
    scaler_memoria_sub.transform(df_obs_sub[numeric_features_memoria])
], axis=1)

processed_actions_sub = scaler_acao_sub.transform(np.array([t.action for t in transitions_sub]).reshape(-1, 1))
processed_rewards_sub = scaler_recompensa_sub.transform(np.array([t.reward for t in transitions_sub]).reshape(-1, 1)).flatten()

# (CÓDIGO NOVO / CORRIGIDO)
print("Convertendo para Episódio Assinatura...")
# Mesma correção: Argumentos posicionais e terminated=False
episode_sub = Episode(
    processed_observations_sub.astype(np.float32),
    processed_actions_sub.astype(np.float32),
    processed_rewards_sub.reshape(-1, 1).astype(np.float32),
    False # terminated=False
)

buffer_sub = ReplayBuffer(FIFOBuffer(limit=len(transitions_sub)), episodes=[episode_sub])
with open('rl_assinatura_buffer.h5', 'w+b') as f:
    buffer_sub.dump(f)

joblib.dump(scaler_memoria_sub, 'scaler_assinatura_memoria.joblib')
joblib.dump(scaler_acao_sub, 'scaler_assinatura_acao.joblib')
joblib.dump(scaler_recompensa_sub, 'scaler_assinatura_recompensa.joblib')

print(f"✓ Buffer Assinatura salvo: {len(transitions_sub)} transições.")

# ============================================================================
# BLOCO 5: RESUMO FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMO FINAL - Geração Dinâmica Concluída")
print("="*80)
print(f"Faixas de Preço Configuradas: {USER_PRICE_RANGES}")
print("Todos os datasets e scalers foram atualizados e escalados proporcionalmente.")
print("Próximo passo: Rodar os notebooks de treino para aprender as novas faixas.")