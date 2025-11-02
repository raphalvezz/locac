import json
import pandas as pd
import numpy as np
import d3rlpy
import uvicorn
import traceback 
import gymnasium as gym

# --- 1. Importações de Configuração ---
from d3rlpy.algos import CQLConfig
from d3rlpy.models import QRQFunctionFactory 
from d3rlpy.dataset import Signature, ReplayBuffer, FIFOBuffer, Episode 

# --- 2. Importações do Servidor ---
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 3. Carregar Configurações (JSON) ---
print("Carregando arquivos de configuração...")
try:
    with open('colunas_observacao.json', 'r') as f:
        observation_cols = json.load(f)

    with open('config_orcamento.json', 'r') as f:
        budget_config = json.load(f)
        bins = budget_config['bins']
        labels = budget_config['labels']
    print("Configurações carregadas.")
except FileNotFoundError as e:
    print(f"!!! ERRO FATAL: Arquivo de configuração não encontrado: {e} !!!")
    print(f"!!! {e.filename} não foi encontrado. Certifique-se de que está na pasta 'backend' !!!")
    exit() 

# --- 4. RE-CRIAR A ARQUITETURA DO MODELO ---
print("Recriando a arquitetura do modelo...")
cql_config = CQLConfig(
    q_func_factory=QRQFunctionFactory(n_quantiles=64),
    batch_size=256,
    n_action_samples=10,
    alpha_learning_rate=1e-4,
    conservative_weight=5.0
)
cql_sac_pricer_iqn = cql_config.create(device='cpu')
print("Arquitetura criada.")

# --- 5. CONSTRUIR O "CASCO" (CORRIGIDO FINAL) ---
print("Construindo o 'casco' do modelo...")
try:
    # 1. Define o formato (shape) dos nossos dados
    observation_shape = (len(observation_cols),) # ex: (28,)
    action_shape = (1,) # ex: (1,)
    
    print("... Criando 'signatures' e buffer vazio...")
    
    # 2. Criar as "Signatures" (formas) manualmente
    #    (O SEU ERRO NA LINHA 68 PROVA QUE ESTE BLOCO ESTÁ FALTANDO)
    obs_signature = Signature(
    shape=[observation_shape],  # ex: [(28,)]
    dtype=[np.float32]
)
    act_signature = Signature(
        shape=[action_shape],  # ex: [(1,)]
        dtype=[np.float32]
    )
    reward_signature = Signature(shape=[()], dtype=[np.float32])

observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)
action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=action_shape, dtype=np.float32)

dummy_buffer = ReplayBuffer(
    FIFOBuffer(limit=10),
    observation_space=observation_space,
    action_space=action_space,
)


    
    # 3. Criar um buffer VAZIO, mas passando as signatures
    #    (Esta é a Linha 76)
    dummy_buffer = ReplayBuffer(
        FIFOBuffer(limit=10), 
        observation_signature=obs_signature, 
        action_signature=act_signature
    ) 
    # --- FIM DA CORREÇÃO ---

    # 4. Criar "dados" falsos para UMA TRANSIÇÃO
    dummy_obs = np.zeros(observation_shape, dtype=np.float32)
    dummy_action = np.zeros(action_shape, dtype=np.float32)
    dummy_reward = 0.0
    dummy_next_obs = np.zeros(observation_shape, dtype=np.float32)
    dummy_terminal = 1.0 

    # 5. Adiciona a transição falsa DIRETAMENTE ao buffer
    dummy_buffer.add_transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=dummy_reward,
        next_observation=dummy_next_obs,
        terminal=dummy_terminal
    )

    # 6. Chama o "build_with_dataset"
    cql_sac_pricer_iqn.build_with_dataset(dummy_buffer)
    # -----------------------------------------------------------------
    
    print("Casco do modelo construído com sucesso.")

except Exception as e:
    print(f"!!! ERRO FATAL ao construir o casco: {e} !!!")
    traceback.print_exc()
    exit()

# --- 6. CARREGAR OS PESOS ---
print("Carregando modelo de RL (pesos treinados)...")
try:
    cql_sac_pricer_iqn.load_model(fname='modelo_rl_final.pt')
    print("SUCESSO: Modelo carregado.")
except Exception as e:
    print(f"!!! ERRO FATAL ao carregar o modelo: {e} !!!")
    print("!!! Verifique se o 'modelo_rl_final.pt' foi baixado corretamente do Colab !!!")
    traceback.print_exc()
    exit()

# --- 7. Definir o Modelo de Entrada da API ---
class SimulationInput(BaseModel):
    region: str
    content: str
    age: str
    gender: str
    platform: str
    budget: float
    product_tier: str

# --- 8. Inicializar o App FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 9. Função de Predição ---
def get_recommendation_from_model(data: SimulationInput):
    try:
        budget_category_obj = pd.cut([data.budget], bins=bins, labels=labels, right=False)[0]
        budget_category_str = str(budget_category_obj)
    except (ValueError, IndexError):
        budget_category_str = labels[0] 

    scenario_data = {
        'Region': [data.region], 
        'Content_Type': [data.content], 
        'Target_Age': [data.age],
        'Target_Gender': [data.gender], 
        'Platform': [data.platform],
        'Budget': [budget_category_str],
        'Product_Tier': [data.product_tier]
    }
    
    # **** ESTA LINHA ESTAVA FALTANDO ANTES ****
    scenario_df = pd.DataFrame(scenario_data)
    # ******************************************

    state_features = ['Region', 'Content_Type', 'Target_Age', 'Target_Gender', 'Platform', 'Budget', 'Product_Tier']
    for col in state_features:
        if col in scenario_df.columns:
            scenario_df[col] = scenario_df[col].astype(str)

    scenario_onehot = pd.get_dummies(scenario_df, columns=state_features)
    observation = scenario_onehot.reindex(columns=observation_cols, fill_value=0).values.astype(np.float32)

    if observation.shape[0] == 1:
        recommended_price_array = cql_sac_pricer_iqn.predict(observation.reshape(1, -1))[0]
        recommended_price = float(recommended_price_array[0]) 
        
        return {
            "recommended_price": recommended_price,
            "estimated_roi": 150.0, # Placeholder
            "var_5_percent": -500.0, # Placeholder
            "cvar_5_percent": -1200.0 # Placeholder
        }
    else:
        raise ValueError("Falha ao processar a observação.")

# --- 10. Endpoint da API ---
@app.post("/api/simulate")
async def simulate_campaign(input_data: SimulationInput):
    try:
        recommendation = get_recommendation_from_model(input_data)
        return recommendation
    except Exception as e:
        print("--- ERRO DURANTE A PREDIÇÃO ---")
        traceback.print_exc()
        print("-------------------------------")
        return {"error": str(e)}, 500

# --- 11. Rodar o servidor ---
if __name__ == "__main__":
    print("Iniciando servidor FastAPI em http://127.0.0.1:8000")
    # ATENÇÃO: Mude a string para o novo nome do arquivo
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
