#!/usr/bin/env python3
"""
Generator - Gêmeo Digital Econômico (v2 - Produtos Digitais)
Gera datasets sintéticos para SL e RL (Venda Única e Assinatura)
Incorpora features de produto digital conforme 'prompt + alterações.pdf'.
"""

import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import d3rlpy
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode
import os
from tqdm import tqdm

print("="*80)
print("GENERATOR (v2 - Produtos Digitais) - Gêmeo Digital Econômico")
print("="*80)

# ============================================================================
# BLOCO 1: Definir o Gêmeo Digital (Economia-Base)
# ============================================================================

print("\n[1/5] Definindo parâmetros econômicos e de produto...")

# Parâmetros Econômicos Base (por Região)
economic_parameters = {
    'North America': {'beta': 0.015, 'c0': 100.0, 'a0': 50.0},
    'Europe':        {'beta': 0.018, 'c0': 80.0,  'a0': 55.0},
    'Asia':          {'beta': 0.012, 'c0': 120.0, 'a0': 45.0},
    'South America': {'beta': 0.020, 'c0': 70.0,  'a0': 40.0},
    'Africa':        {'beta': 0.022, 'c0': 60.0,  'a0': 35.0},
}

# Parâmetros de Custo Base (CPAs) por região x tipo de produto/tier
cost_parameters = {
    'North America': {
        'InfoProduto_Low': {'cpa_venda': 15, 'cpa_assinatura': 40},
        'InfoProduto_High': {'cpa_venda': 40, 'cpa_assinatura': 70},
        'SaaS_Low': {'cpa_venda': 20, 'cpa_assinatura': 50},
        'SaaS_High': {'cpa_venda': 50, 'cpa_assinatura': 90},
        'Serviço_Low': {'cpa_venda': 25, 'cpa_assinatura': 60},
        'Serviço_High': {'cpa_venda': 70, 'cpa_assinatura': 120},
    },
    'Europe': {
        'InfoProduto_Low': {'cpa_venda': 18, 'cpa_assinatura': 45},
        'InfoProduto_High': {'cpa_venda': 45, 'cpa_assinatura': 75},
        'SaaS_Low': {'cpa_venda': 22, 'cpa_assinatura': 55},
        'SaaS_High': {'cpa_venda': 55, 'cpa_assinatura': 100},
        'Serviço_Low': {'cpa_venda': 28, 'cpa_assinatura': 65},
        'Serviço_High': {'cpa_venda': 75, 'cpa_assinatura': 130},
    },
    'Asia': {
        'InfoProduto_Low': {'cpa_venda': 12, 'cpa_assinatura': 35},
        'InfoProduto_High': {'cpa_venda': 35, 'cpa_assinatura': 60},
        'SaaS_Low': {'cpa_venda': 15, 'cpa_assinatura': 40},
        'SaaS_High': {'cpa_venda': 40, 'cpa_assinatura': 80},
        'Serviço_Low': {'cpa_venda': 20, 'cpa_assinatura': 50},
        'Serviço_High': {'cpa_venda': 60, 'cpa_assinatura': 110},
    },
    'South America': {
        'InfoProduto_Low': {'cpa_venda': 14, 'cpa_assinatura': 38},
        'InfoProduto_High': {'cpa_venda': 38, 'cpa_assinatura': 65},
        'SaaS_Low': {'cpa_venda': 18, 'cpa_assinatura': 45},
        'SaaS_High': {'cpa_venda': 45, 'cpa_assinatura': 85},
        'Serviço_Low': {'cpa_venda': 22, 'cpa_assinatura': 55},
        'Serviço_High': {'cpa_venda': 65, 'cpa_assinatura': 115},
    },
    'Africa': {
        'InfoProduto_Low': {'cpa_venda': 10, 'cpa_assinatura': 30},
        'InfoProduto_High': {'cpa_venda': 30, 'cpa_assinatura': 55},
        'SaaS_Low': {'cpa_venda': 12, 'cpa_assinatura': 35},
        'SaaS_High': {'cpa_venda': 35, 'cpa_assinatura': 70},
        'Serviço_Low': {'cpa_venda': 18, 'cpa_assinatura': 45},
        'Serviço_High': {'cpa_venda': 50, 'cpa_assinatura': 100},
    },
}

# Features categóricas (contexto + produto digital)
feature_values = {
    'Regiao': ['North America', 'Europe', 'Asia', 'South America', 'Africa'],
    'Plataforma': ['Instagram', 'Facebook', 'LinkedIn', 'Twitter', 'Pinterest'],
    'Tier': ['Low Ticket', 'High Ticket'],
    'Idade': ['18-24', '25-34', '35-44', '45-54', '55+'],
    'Genero': ['Female', 'Male', 'Other'],
    'Conteudo': ['Image', 'Video', 'Text', 'Carousel'],
    # Produto digital
    'Tipo_Produto': ['SaaS', 'InfoProduto', 'Serviço'],
    'Modelo_Cobranca': ['Assinatura', 'Venda Unica'],
    'Complexidade_Oferta': ['Baixa', 'Media', 'Alta'],
}

categorical_features = [
    'Regiao', 'Plataforma', 'Tier', 'Idade', 'Genero', 'Conteudo',
    'Tipo_Produto', 'Modelo_Cobranca', 'Complexidade_Oferta'
]

# Feature numérica de estado base
numeric_features_base = ['Orcamento']

# Features numéricas de memória (apenas Assinatura)
numeric_features_memoria = [
    'dias_desde_ultima_interacao',
    'clv_estimate_percentile',
    'avg_price_offered_segment_90d',
    'price_volatility_30d',
]

colunas_estado_base = None
colunas_estado_assinatura = None

# ----------------- Funções do gêmeo digital -----------------

def get_demand_modifiers(estado):
    """Fatores de ajuste de demanda com base no tipo de produto e complexidade."""
    c0_factor = 1.0
    beta_factor = 1.0

    # Tipo de Produto
    if estado['Tipo_Produto'] == 'SaaS':
        c0_factor *= 1.2
        beta_factor *= 0.9
    elif estado['Tipo_Produto'] == 'InfoProduto':
        c0_factor *= 1.0
        beta_factor *= 1.1
    elif estado['Tipo_Produto'] == 'Serviço':
        c0_factor *= 0.8
        beta_factor *= 1.0

    # Complexidade
    if estado['Complexidade_Oferta'] == 'Alta':
        c0_factor *= 0.7
        beta_factor *= 0.8
    elif estado['Complexidade_Oferta'] == 'Media':
        c0_factor *= 1.0
    elif estado['Complexidade_Oferta'] == 'Baixa':
        c0_factor *= 1.3
        beta_factor *= 1.2

    return c0_factor, beta_factor


def calculate_demand(estado, preco):
    """Demanda (nº de conversões) em função de estado + preço."""
    regiao = estado['Regiao']
    params = economic_parameters.get(regiao, economic_parameters['North America'])

    c0_base = params['c0']
    beta_base = params['beta']
    a0 = params['a0']

    c0_factor, beta_factor = get_demand_modifiers(estado)

    c0_final = c0_base * c0_factor
    beta_final = beta_base * beta_factor

    # Efeito do orçamento
    orcamento_factor = np.log1p(estado['Orcamento']) / np.log1p(5000)
    c0_final *= (0.5 + orcamento_factor)

    conversoes = c0_final * np.exp(-beta_final * (preco - a0))
    return max(0.0, conversoes)


def calculate_churn_rate(estado, preco_mensalidade):
    """Taxa de churn mensal em função de preço + memória + complexidade."""
    churn_base = 0.05  # 5% base
    price_factor = (preco_mensalidade / 100.0) * 0.02

    # Âncora de preço
    anchor_price = estado.get('avg_price_offered_segment_90d', preco_mensalidade)
    anchor_diff = (preco_mensalidade - anchor_price) / max(anchor_price, 1e-8)
    anchor_factor = max(0.0, anchor_diff) * 0.1

    # Tempo sem interação
    dias = estado.get('dias_desde_ultima_interacao', 30.0)
    dias_factor = (dias / 90.0) * 0.05

    # Complexidade
    if estado['Complexidade_Oferta'] == 'Alta':
        complex_factor = 0.03
    elif estado['Complexidade_Oferta'] == 'Media':
        complex_factor = 0.01
    else:
        complex_factor = 0.0

    churn_final = churn_base + price_factor + anchor_factor + dias_factor + complex_factor
    return float(min(max(churn_final, 0.01), 0.8))


def get_cpa(estado, tipo_cpa='cpa_venda'):
    """Retorna CPA (venda ou assinatura) com fallback robusto."""
    try:
        regiao = estado['Regiao']
        chave_produto = f"{estado['Tipo_Produto']}_{'Low' if estado['Tier'] == 'Low Ticket' else 'High'}"

        if regiao not in cost_parameters:
            regiao = 'North America'
        if chave_produto not in cost_parameters[regiao]:
            chave_produto = 'InfoProduto_Low'

        return cost_parameters[regiao][chave_produto][tipo_cpa]
    except Exception:
        return 20.0 if tipo_cpa == 'cpa_venda' else 50.0


def get_reward_venda_unica(estado, preco):
    """Lucro de venda pontual."""
    cpa = get_cpa(estado, 'cpa_venda')
    conversoes = calculate_demand(estado, preco)
    lucro = conversoes * (preco - cpa)
    return float(max(0.0, lucro))


def get_reward_assinatura(estado, preco_mensalidade):
    """LTV aproximado para assinatura."""
    cpa = get_cpa(estado, 'cpa_assinatura')
    churn = calculate_churn_rate(estado, preco_mensalidade)
    if churn <= 0.0:
        return 0.0
    ltv = (preco_mensalidade / churn) - cpa
    return float(max(0.0, ltv))


print(f"✓ Regiões: {feature_values['Regiao']}")
print(f"✓ Custos definidos para {len(cost_parameters)} regiões e {len(cost_parameters['North America'])} tipos de produto.")
print("✓ Funções de demanda, churn e lucro modificadas (v2) implementadas.")
print(f"✓ {len(categorical_features) + len(numeric_features_base)} features de estado base definidas.")

# Estrutura de Transição simples para organizar dados
class Transition:
    def __init__(self, observation, action, reward, terminal):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal

# ============================================================================
# BLOCO 2: Gerar Dataset Supervised Learning (SL)
# ============================================================================

print("\n[2/5] Gerando dataset Supervised Learning...")

def generate_sl_dataset(num_samples=50000):
    data = []

    for _ in tqdm(range(num_samples)):
        # Estado básico
        estado = {feat: np.random.choice(feature_values[feat]) for feat in categorical_features}
        estado['Orcamento'] = np.random.uniform(500, 10000)

        # Lógica de preço/lucro por modelo de cobrança
        if estado['Modelo_Cobranca'] == 'Venda Unica':
            cpa = get_cpa(estado, 'cpa_venda')
            margem = np.random.uniform(10, 150)
            preco_amostra = cpa + margem
            lucro_real = get_reward_venda_unica(estado, preco_amostra)
        else:
            cpa = get_cpa(estado, 'cpa_assinatura')
            margem = np.random.uniform(5, 50)
            preco_amostra = cpa + margem

            # Features de memória artificiais só para cálculo de LTV
            estado['dias_desde_ultima_interacao'] = np.random.uniform(1, 90)
            estado['clv_estimate_percentile'] = np.random.uniform(0.1, 1.0)
            estado['avg_price_offered_segment_90d'] = preco_amostra * np.random.uniform(0.8, 1.2)
            estado['price_volatility_30d'] = np.random.uniform(0.5, 10.0)

            lucro_real = get_reward_assinatura(estado, preco_amostra)

        row = estado.copy()
        row['Preco_Amostra'] = preco_amostra
        row['Lucro_Real'] = lucro_real
        data.append(row)

    df = pd.DataFrame(data)
    # SL não precisa das colunas de memória explícitas
    df = df.drop(columns=numeric_features_memoria, errors='ignore')
    return df

df_sl = generate_sl_dataset()
df_sl.to_csv('sl_dataset_combined.csv', index=False)

print(f"✓ Dataset SL salvo: {len(df_sl)} linhas")
print(f"  Lucro médio (SL): ${df_sl['Lucro_Real'].mean():.2f}")

# Pré-processadores SL (simples: OHE categóricas + scaler de orçamento)
ohe_sl = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df_sl[categorical_features])
scaler_sl = StandardScaler().fit(df_sl[numeric_features_base])

joblib.dump(ohe_sl, 'sl_ohe_encoder.joblib')
joblib.dump(scaler_sl, 'sl_scaler_estado.joblib')

print("✓ Scalers SL salvos: 2 arquivos .joblib")

# ============================================================================
# BLOCO 3: Gerar Buffer RL Venda Única (Fixo)
# ============================================================================

print("\n[3/5] Gerando buffer RL Venda Única...")

def generate_rl_offline_buffer(num_samples=100000):
    transitions_rl = []

    for _ in tqdm(range(num_samples)):
        estado = {feat: np.random.choice(feature_values[feat]) for feat in categorical_features}
        estado['Modelo_Cobranca'] = 'Venda Unica'  # força venda única
        estado['Orcamento'] = np.random.uniform(500, 10000)

        cpa = get_cpa(estado, 'cpa_venda')
        margem = np.random.uniform(10, 150)
        preco_amostra = cpa + margem

        recompensa = get_reward_venda_unica(estado, preco_amostra)

        transitions_rl.append(
            Transition(
                observation=estado,
                action=preco_amostra,
                reward=recompensa,
                terminal=0.0
            )
        )

    return transitions_rl

transitions_rl = generate_rl_offline_buffer()
print(f"✓ Buffer RL gerado: {len(transitions_rl)} transições")
print(f"  Recompensa média (RL Venda): ${np.mean([t.reward for t in transitions_rl]):.2f}")

# Pré-processar RL Venda Única
df_obs_rl = pd.DataFrame([t.observation for t in transitions_rl])

ohe_rl = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(df_obs_rl[categorical_features])
scaler_estado_rl = StandardScaler().fit(df_obs_rl[numeric_features_base])
scaler_acao_rl = StandardScaler().fit(
    np.array([t.action for t in transitions_rl]).reshape(-1, 1)
)
scaler_recompensa_rl = StandardScaler().fit(
    np.array([t.reward for t in transitions_rl]).reshape(-1, 1)
)

colunas_estado_base = list(ohe_rl.get_feature_names_out()) + numeric_features_base
with open('colunas_estado_base.json', 'w') as f:
    json.dump(colunas_estado_base, f)

# Aplicar transformações
ohe_data_rl = ohe_rl.transform(df_obs_rl[categorical_features])
scaled_data_rl = scaler_estado_rl.transform(df_obs_rl[numeric_features_base])
processed_observations = np.concatenate([ohe_data_rl, scaled_data_rl], axis=1)

processed_actions = scaler_acao_rl.transform(
    np.array([t.action for t in transitions_rl]).reshape(-1, 1)
)
processed_rewards = scaler_recompensa_rl.transform(
    np.array([t.reward for t in transitions_rl]).reshape(-1, 1)
).flatten()

for i in range(len(transitions_rl)):
    transitions_rl[i].observation = processed_observations[i]
    transitions_rl[i].action = processed_actions[i]
    transitions_rl[i].reward = processed_rewards[i]

print("Convertendo transições RL Venda para formato Episode...")

print("Convertendo transições RL Venda para formato Episode...")

# Arrays já processados
obs_array = processed_observations.astype("float32")  # shape: (N, obs_dim)
act_array = processed_actions.astype("float32")       # shape: (N, 1)
rew_array = processed_rewards.astype("float32")       # shape: (N,)

episodes = []

for i in range(len(rew_array)):
    obs = obs_array[i]   # (obs_dim,)
    act = act_array[i]   # (1,)
    rew = rew_array[i]   # escalar

    # Episódio de 1 passo: [s, s] como obs_t e obs_{t+1}
    ep_obs = np.stack([obs, obs], axis=0)          # (2, obs_dim)
    ep_act = act.reshape(1, -1)                    # (1, 1)
    ep_rew = np.array([rew], dtype=np.float32)     # (1,)

    episode = Episode(
        observations=ep_obs,
        actions=ep_act,
        rewards=ep_rew,
        terminated=True,   # um episódio de 1 transição
    )
    episodes.append(episode)

buffer = ReplayBuffer(FIFOBuffer(limit=len(episodes)), episodes=episodes)
buffer.dump("rl_offline_buffer.h5")
print("✓ Buffer RL Venda salvo em 'rl_offline_buffer.h5'")
print(f"✓ Buffer salvo: rl_offline_buffer.h5 ({buffer.size()} transições)")



joblib.dump(ohe_rl, 'ohe_encoder.joblib')
joblib.dump(scaler_estado_rl, 'scaler_estado.joblib')
joblib.dump(scaler_acao_rl, 'scaler_acao.joblib')
joblib.dump(scaler_recompensa_rl, 'scaler_recompensa.joblib')

print(f"✓ Buffer salvo: rl_offline_buffer.h5 ({buffer.size()} transições)")
print("✓ Scalers RL Venda salvos: 4 arquivos .joblib")

# ============================================================================
# BLOCO 4: Gerar Buffer RL Assinatura
# ============================================================================

print("\n[4/5] Gerando buffer RL Assinatura...")

def generate_rl_assinatura_buffer(num_samples=50000):
    transitions_sub = []

    for _ in tqdm(range(num_samples)):
        estado = {feat: np.random.choice(feature_values[feat]) for feat in categorical_features}
        estado['Modelo_Cobranca'] = 'Assinatura'
        estado['Orcamento'] = np.random.uniform(500, 10000)

        # memória
        estado['dias_desde_ultima_interacao'] = np.random.uniform(1, 90)
        estado['clv_estimate_percentile'] = np.random.uniform(0.1, 1.0)
        estado['avg_price_offered_segment_90d'] = np.random.uniform(20, 100)
        estado['price_volatility_30d'] = np.random.uniform(0.5, 10.0)

        cpa = get_cpa(estado, 'cpa_assinatura')
        margem = np.random.uniform(5, 50)
        preco_amostra = cpa + margem

        recompensa = get_reward_assinatura(estado, preco_amostra)

        transitions_sub.append(
            Transition(
                observation=estado,
                action=preco_amostra,
                reward=recompensa,
                terminal=0.0
            )
        )

    return transitions_sub

transitions_sub = generate_rl_assinatura_buffer()
print(f"✓ Buffer Assinatura gerado: {len(transitions_sub)} transições")
print(f"  Recompensa média (LTV): ${np.mean([t.reward for t in transitions_sub]):.2f}")

# Pré-processar RL Assinatura
df_obs_sub = pd.DataFrame([t.observation for t in transitions_sub])

# Reusa OHE e scaler de estado base do RL fixo
ohe_base = ohe_rl
scaler_estado_base = scaler_estado_rl

scaler_memoria_sub = StandardScaler().fit(df_obs_sub[numeric_features_memoria])
scaler_acao_sub = StandardScaler().fit(
    np.array([t.action for t in transitions_sub]).reshape(-1, 1)
)
scaler_recompensa_sub = StandardScaler().fit(
    np.array([t.reward for t in transitions_sub]).reshape(-1, 1)
)

colunas_estado_assinatura = colunas_estado_base + numeric_features_memoria
with open('colunas_estado_assinatura.json', 'w') as f:
    json.dump(colunas_estado_assinatura, f)

ohe_data_sub = ohe_base.transform(df_obs_sub[categorical_features])
scaled_data_base_sub = scaler_estado_base.transform(df_obs_sub[numeric_features_base])
scaled_data_memoria_sub = scaler_memoria_sub.transform(df_obs_sub[numeric_features_memoria])

processed_observations_sub = np.concatenate(
    [ohe_data_sub, scaled_data_base_sub, scaled_data_memoria_sub],
    axis=1
)

processed_actions_sub = scaler_acao_sub.transform(
    np.array([t.action for t in transitions_sub]).reshape(-1, 1)
)
processed_rewards_sub = scaler_recompensa_sub.transform(
    np.array([t.reward for t in transitions_sub]).reshape(-1, 1)
).flatten()

for i in range(len(transitions_sub)):
    transitions_sub[i].observation = processed_observations_sub[i]
    transitions_sub[i].action = processed_actions_sub[i]
    transitions_sub[i].reward = processed_rewards_sub[i]

print("Convertendo transições Assinatura para formato Episode...")

obs_array_sub = processed_observations_sub.astype("float32")  # (N, obs_dim_sub)
act_array_sub = processed_actions_sub.astype("float32")       # (N, 1)
rew_array_sub = processed_rewards_sub.astype("float32")       # (N,)

episodes_sub = []

for i in range(len(rew_array_sub)):
    obs = obs_array_sub[i]
    act = act_array_sub[i]
    rew = rew_array_sub[i]

    ep_obs = np.stack([obs, obs], axis=0)          # (2, obs_dim_sub)
    ep_act = act.reshape(1, -1)                    # (1, 1)
    ep_rew = np.array([rew], dtype=np.float32)     # (1,)

    episode = Episode(
        observations=ep_obs,
        actions=ep_act,
        rewards=ep_rew,
        terminated=True,
    )
    episodes_sub.append(episode)

buffer_sub = ReplayBuffer(FIFOBuffer(limit=len(episodes_sub)), episodes=episodes_sub)
buffer_sub.dump("rl_assinatura_buffer.h5")
print("✓ Buffer RL Assinatura salvo em 'rl_assinatura_buffer.h5'")
print(f"✓ Buffer salvo: rl_assinatura_buffer.h5 ({buffer_sub.size()} transições)")




joblib.dump(scaler_memoria_sub, 'scaler_assinatura_memoria.joblib')
joblib.dump(scaler_acao_sub, 'scaler_assinatura_acao.joblib')
joblib.dump(scaler_recompensa_sub, 'scaler_assinatura_recompensa.joblib')

print(f"✓ Buffer salvo: rl_assinatura_buffer.h5 ({buffer_sub.size()} transições)")
print("✓ Scalers Assinatura salvos: 3 arquivos .joblib (reutilizando OHE e scaler de estado base)")

# ============================================================================
# BLOCO 5: RESUMO FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMO FINAL - Artefatos Gerados")
print("="*80)

arquivos = [
    ('sl_dataset_combined.csv',          'Dataset SL'),
    ('rl_offline_buffer.h5',             'Buffer RL Venda Única'),
    ('rl_assinatura_buffer.h5',          'Buffer RL Assinatura'),
    ('ohe_encoder.joblib',               'Encoder Categórico (OHE RL)'),
    ('scaler_estado.joblib',             'Scaler Estado (RL Base)'),
    ('scaler_acao.joblib',               'Scaler Ação (RL Venda)'),
    ('scaler_recompensa.joblib',         'Scaler Recompensa (RL Venda)'),
    ('scaler_assinatura_memoria.joblib', 'Scaler Memória (Assinatura)'),
    ('scaler_assinatura_acao.joblib',    'Scaler Ação (Assinatura)'),
    ('scaler_assinatura_recompensa.joblib', 'Scaler Recompensa (Assinatura)'),
    ('colunas_estado_base.json',         'Schema Estado Base'),
    ('colunas_estado_assinatura.json',   'Schema Estado Assinatura'),
    ('sl_ohe_encoder.joblib',            'Encoder SL (OHE)'),
    ('sl_scaler_estado.joblib',          'Scaler SL Estado'),
]

for arquivo, descricao in arquivos:
    if os.path.exists(arquivo):
        tamanho = os.path.getsize(arquivo) / (1024 * 1024)
        print(f"✓ {descricao:<35} {arquivo:<35} {tamanho:>8.2f} MB")
    else:
        print(f"✗ {descricao:<35} {arquivo:<35} FALTANDO")

print("="*80)
print("\n🎉 SUCESSO! Todos os datasets foram gerados (SL, RL Venda, RL Assinatura).")
print("🚀 Próximo passo: rodar SL_FINAL, RL_OFF e RL_Assinatura com esses artefatos.")
