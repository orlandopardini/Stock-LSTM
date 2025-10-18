# Stock LSTM

Este repositório implementa um **pipeline completo de previsão de séries financeiras** com:
- **API** (Flask) para servir dados, previsões, backtests e tarefas de atualização/treino;
- **Camada de treino/avaliação** (TensorFlow/Keras + scikit‑learn);
- **Persistência** (SQLite via SQLAlchemy) de preços, métrica de modelos e registro de “vencedor”;
- **Monitoramento** (Prometheus) e **admin tasks** protegidas por `X-API-KEY`;
- **Frontend** (HTML/Plotly) com painéis interativos, tema escuro e bordas neon;
- **Empacotamento** para **Render.com** com `gunicorn`, volume persistente de modelos e cron diário.

> **Status**: pronto para rodar localmente e para deploy no Render. O banco SQLite e o diretório `models/` devem ser persistidos em produção.

<img width="1886" height="949" alt="image" src="https://github.com/user-attachments/assets/9095e0a1-cc25-44fe-a29f-7db342a48f4f" />
<img width="1885" height="678" alt="image" src="https://github.com/user-attachments/assets/339f1315-1796-4aa6-ac6c-b5136b68a933" />
<img width="1889" height="483" alt="image" src="https://github.com/user-attachments/assets/0388f1fd-c2c3-40d9-acb7-b74d6aa8cd02" />

---

## Sumário
1. [Arquitetura geral](#arquitetura-geral)
2. [Módulos & responsabilidades](#módulos--responsabilidades)
3. [Modelos e treinamento](#modelos-e-treinamento)
4. [Banco de dados](#banco-de-dados)
5. [API — contratos principais](#api--contratos-principais)
6. [Frontend (Plotly) & UX](#frontend-plotly--ux)
7. [Monitoramento e métricas](#monitoramento-e-métricas)
8. [Execução local](#execução-local)
9. [Variáveis de ambiente](#variáveis-de-ambiente)
10. [Deploy (Render.com)](#deploy-rendercom)
11. [Tarefas/cron & automação](#tarefascron--automação)
12. [Requisitos (requirements.txt)](#requisitos-requirementstxt)
13. [Dicas de produção](#dicas-de-produção)
14. [Troubleshooting rápido](#troubleshooting-rápido)

---

## Execução local

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export FLASK_APP=wsgi.py               # Windows: set FLASK_APP=wsgi.py
flask run                              # http://127.0.0.1:5000
```

- **Swagger**: `http://127.0.0.1:5000/apidocs`
- **Site**: `http://127.0.0.1:5000/` (gráficos)
- **Seed/simulação**: `http://127.0.0.1:5000/simulate`

---

## Arquitetura geral

```
wsgi.py                  # ponto de entrada WSGI (Gunicorn) → create_app()
app/
  __init__.py            # app factory; DB; Swagger; blueprints; /metrics
  api.py                 # rotas REST (series, backtest, predict, models, tasks)
  web.py                 # páginas HTML (index/simulate) + assets estáticos
  models.py              # ORM (SQLAlchemy) — tabelas e índices
  monitoring.py          # /metrics Prometheus + middleware de latência
  monitoring_simple.py   # alternativa /metrics com psutil (CPU/RAM)
  timing.py              # utilitário simples de cronometria (se aplicável)
ml/
  constants.py           # hiperparâmetros e constantes globais
  data.py                # ingest, limpeza, features e splits
  model_zoo.py           # arquiteturas de modelos (LSTM etc.)
  trainer.py             # ciclo de treino (fit) e salvamento
  eval.py                # métricas, backtests e validações
  pipeline.py            # orquestração: baixar dados → treinar → registrar
  monitor.py             # hooks de treinamento + registro de métricas
static/
  index.html, simulate.html, style.css, index.js, bootstrap.min.css
models/                  # modelos salvos (.keras), scalers, artefatos
instance/app.db          # SQLite (criado em runtime)
render.yaml              # definição de serviço e cron para Render.com
README.md
```

> O projeto usa **app factory** para isolar configuração e facilitar testes/deploy.
> Os módulos de ML ficam separados para manter a API enxuta e a orquestração clara.

---

## Módulos & responsabilidades

### Camada Web / API (`app/`)

- **`__init__.py`**
  - Cria a aplicação Flask; configura **SQLAlchemy** (SQLite em `instance/app.db`), **Migrate**, **Swagger** (Flasgger).
  - Registra **blueprints** (`api`, `web`) e expõe `/metrics` via `monitoring.py` ou `monitoring_simple.py`.
  - Ativa PRAGMAs do SQLite (WAL/timeout) para melhor confiabilidade.

- **`api.py`**
  - Endpoints REST:
    - `GET /api/tickers` — lista de tickers suportados.
    - `GET /api/series?ticker=...&limit=...` — série histórica (data/close).
    - `GET /api/backtest?ticker=...&window=...` — dados de backtest do modelo atual.
    - `GET /api/predict?ticker=...` — previsão do próximo dia útil.
    - `GET /api/models/best?ticker=...` — metadados e métricas do “vencedor”.
    - `GET /api/models/summary?ticker=...` — top N modelos para o ticker.
    - `POST /api/tasks/daily_update` — **tarefa protegida** (`X-API-KEY`) para atualizar dados/treinar.
  - Integra-se com a camada ML/pipeline e com o ORM para consultar/registrar resultados.

- **`web.py`**
  - Renderiza **`index.html`** (dashboard) e **`simulate.html`** (preenchimento/seed de dados).
  - Fornece estáticos: `style.css`, `index.js`, `bootstrap.min.css`.

- **`monitoring.py` / `monitoring_simple.py`**
  - `/metrics` no formato **Prometheus**.
  - Versão completa adiciona middleware de **latência por rota** e contadores de requests.
  - Versão simples expõe gauges de **CPU/RAM** via `psutil`.

- **`wsgi.py`**
  - Expõe `app = create_app()` para o **Gunicorn**.

### Camada de ML (`ml/`)

- **`constants.py`**: hiperparâmetros padrão (janelas, épocas, batch size), nomes de colunas, seeds, etc.
- **`data.py`**: download com **yfinance**, limpeza, engenharia de atributos (lags, returns, normalização), splits treino/validação/teste.
- **`model_zoo.py`**: definição de arquiteturas; principal é **LSTM** (com camadas empilhadas + dense “skip” opcional). Pode expor funções `build_*` que recebem `input_shape` e `hyperparams`.
- **`trainer.py`**: rotina de treino (**fit**), callbacks (EarlyStopping/ReduceLROnPlateau), salvamento de artefatos (modelo `.keras`, `scaler.pkl`).
- **`eval.py`**: cálculo de métricas (**RMSE**, **MAE**, **MAPE**, **R²**, **accuracy** categórica opcional), backtest e geração de séries previstas.
- **`pipeline.py`**: orquestra: **coleta** → **feature** → **treina** → **avalia** → **registra**; escreve resultados no **ORM** e move o vencedor para `ModelRegistry (is_winner=1)`.
- **`monitor.py`**: hooks para registrar métricas/tempos do pipeline no Prometheus (se habilitado).

> A camada ML é **agnóstica** do Flask; a API a utiliza como biblioteca. Isso facilita testes offline e CLIs futuros.

---

## Modelos e treinamento

- **Família principal**: **LSTM** para séries temporais de preço de fechamento (Close). Arquitetura típica:
  - 1–2 camadas LSTM (64/64) → Dense final; ativação linear; perda MSE/MAE;
  - Entrada: janelas deslizantes de tamanho `WINDOW` (ex.: 60/90);
  - Normalização com `StandardScaler/MinMaxScaler` aplicada por **ticker**.
- **Validação/backtest**:
  - Janela “rolling” para avaliação recente;
  - Métricas: RMSE, MAE, MAPE, R² (contínuo) e ACC (se houver discretização de direção).
- **Seleção de vencedor**:
  - O pipeline salva múltiplos modelos e registra **`is_winner=1`** em `ModelRegistry` para o melhor (menor MAE/RMSE em janela-alvo).
- **Artefatos**:
  - Modelos Keras (`.keras`), scaler (`.pkl`) e JSON de hiperparâmetros/score por versão.

**Observações práticas**:
- **1 worker** de Gunicorn recomendado em produção (TensorFlow consome memória). Use threads para concorrência leve.
- Fixe seeds apenas se aceitar custo/variância (TF pode variar entre builds; `TF_ENABLE_ONEDNN_OPTS=0` ajuda reprodutibilidade).

---

## Banco de dados

### Tabelas principais (SQLAlchemy)

- **`PrecoDiario`**: preços históricos por `(ticker, date)`; índice único; campos de OHLC/volume (pelo menos `close`).
- **`ResultadoMetricas`**: métricas de execução (RMSE/MAE/MAPE/R²/ACC, janela, data, versão do modelo).
- **`ModelRegistry`**: catálogo de modelos treinados por ticker, com `is_winner` e metadados (`model_name`, `version`, `registered_at`). 
- **`RetrainHistory`**: histórico de retreinamentos (quando, quanto tempo, status, exceções).

### Ciclo de dados
1. `/api/tasks/daily_update` coleta **yfinance**, atualiza `PrecoDiario`, treina e registra resultados.
2. `/api/models/best` lê `ModelRegistry` e retorna o vencedor + métricas agregadas.
3. `/api/backtest` consulta previsões recentes + série real para compor gráficos.
4. `/api/series` dá acesso rápido à série para gráficos/clients.

> O schema é criado automaticamente na primeira execução (app factory). Em produção, **persista** a pasta `instance/` para manter o SQLite entre deploys.

---

## API — contratos principais

### `GET /api/tickers`
- **Resposta**: `["AAPL", "NVDA", "MSFT", ...]`

### `GET /api/series?ticker=...&limit=800`
- **Resposta**: 
  ```json
  { "data": [ { "date": "YYYY-MM-DD", "close": 123.45 }, ... ] }
  ```

### `GET /api/backtest?ticker=...&window=180`
- **Resposta (ex.)**:
  ```json
  {
    "dates": ["YYYY-MM-DD", ...],
    "y_true": [123.4, ...],
    "y_pred": [122.9, ...]
  }
  ```

### `GET /api/predict?ticker=...`
- **Resposta (ex.)**:
  ```json
  { "date_next": "YYYY-MM-DD", "pred": 125.67 }
  ```

### `GET /api/models/best?ticker=...`
- **Resposta (ex.)**:
  ```json
  {
    "model_id": 2,
    "model_name": "LSTM(64/64)+Dense",
    "version": "AAPL_2_20251008_180812",
    "registered_at": "2025-10-08T18:08:12",
    "metrics": { "rmse": 9.0532, "mae": 7.1007, "r2": 0.861, "accuracy": 0.466 }
  }
  ```

### `GET /api/models/summary?ticker=...`
- **Resposta**: lista de modelos com campos acima + `is_winner`.

### `POST /api/tasks/daily_update`
- **Headers**: `X-API-KEY: <API_KEY>`  
- **Body**: `{"ticker":"AAPL"}` (padrão)  
- **Efeito**: coleta dados, treina/avalia, registra vencedor. Retorna JSON de status.

> **Segurança**: em produção, mantenha `API_KEY` secreto e **não defina** `DISABLE_API_KEY`. Em dev/local, você pode usar `DISABLE_API_KEY=1` para facilitar.

---

## Frontend (Plotly) & UX

- **Páginas**: `index.html` (painel) e `simulate.html` (seed).  
- **Gráficos principais**:
  - **Preço — Real vs. Previsto (30 dias)**: foco curto com banda de média±DP e **ponto vermelho** do próximo dia (se houver).
  - **Erro (|Real−Prev|) — MAE rolling(20)** + **dispersão y vs ŷ** com legenda explicativa (bem previsto/aceitável/fora).
  - **Série de Fechamento (365 dias)**: janela longa da série e (se disponível) previsto com faixa de distância.
- **Visual**: tema escuro coerente; botões e cards com **borda neon**; toolbar alinhada; logs podem ocupar a linha inteira.
- **Acessibilidade**: legendas centralizadas, títulos claros e números com `tabular-nums` para melhor leitura.

---

## Monitoramento e métricas

- Endpoint **`/metrics`** (Prometheus):
  - Versão completa (`monitoring.py`): middleware de **latência por rota**, **requests em progresso** e contadores por status.
  - Versão simples (`monitoring_simple.py`): **CPU/RAM** via `psutil` — útil em ambientes restritos.
- Integra com **Grafana** facilmente: adicione o serviço Prometheus e aponte o target.

---

## Variáveis de ambiente

| Variável                | Padrão   | Descrição |
|-------------------------|----------|-----------|
| `SECRET_KEY`            | —        | Chave Flask. |
| `API_KEY`               | —        | Chave para `X-API-KEY` nas tasks/rotas sensíveis. |
| `MODELS_DIR`            | `models` | Diretório onde salvar/ler modelos/scalers. |
| `DISABLE_API_KEY`       | `0`      | (Dev) Se `1`, desabilita checagem de API key em localhost. |
| `TF_ENABLE_ONEDNN_OPTS` | —        | Se `0`, reduz variações numéricas do TF. |
| `FLASK_ENV`             | —        | `development` para reloader/stacktraces. |

---

## Deploy (Render.com)

`render.yaml` recomendado:

```yaml
services:
  - type: web
    name: stock-lstm-flask
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn -b 0.0.0.0:$PORT wsgi:app --workers=1 --threads=4 --timeout=120"
    envVars:
      - key: SECRET_KEY
        generateValue: true
      - key: API_KEY
        value: change-me
      - key: MODELS_DIR
        value: models
      # - key: TF_ENABLE_ONEDNN_OPTS
      #   value: "0"
    disk:
      name: models-disk
      mountPath: /opt/render/project/src/models
      sizeGB: 1
    extraDisks:
      - name: instance-disk
        mountPath: /opt/render/project/src/instance
        sizeGB: 1

cronJobs:
  - name: daily-update-train
    schedule: "0 9 * * *"   # UTC (06:00 BRT)
    command: >
      curl -s -X POST "$RENDER_EXTERNAL_URL/api/tasks/daily_update"
      -H "X-API-KEY: $API_KEY"
      -H "Content-Type: application/json"
      -d '{"ticker":"AAPL"}'
```

**Notas**:
- Gunicorn deve **bindar no `$PORT`** do Render.
- Persista **`models/`** e **`instance/`** (SQLite).
- Cron executa em **UTC**.

---

## Tarefas/cron & automação

- **Atualização diária**: `POST /api/tasks/daily_update` (com `X-API-KEY`).
- Possível extensão: tarefas para backfills, limpeza de modelos antigos e geração de relatórios.

Exemplo manual:
```bash
curl -X POST "http://localhost:5000/api/tasks/daily_update" \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL"}'
```

---

## Dicas de produção
- **1 worker** Gunicorn é o mais seguro com TensorFlow (memória); ajuste `threads` para concorrência leve.
- Monitore **latência de rotas** e uso de CPU/RAM; alerte sobre erros no cron.
- Persistência de **SQLite** e **modelos** é obrigatória para não perder histórico/treinos.
- Em ambientes sem AVX/OneDNN, considere `TF_ENABLE_ONEDNN_OPTS=0` para consistência numérica.

---

## Troubleshooting rápido

- **“App sobe mas não responde no Render”** → verifique `startCommand` com `-b 0.0.0.0:$PORT`.
- **Banco “zera” após deploy** → faltou persistir `instance/`.
- **/metrics 404** → conferir se está usando `monitoring.py` (ou registrando o simples).
- **Falha nas tarefas (401)** → faltou header `X-API-KEY`.
- **Previsões inconsistentes** → confira scaler por ticker, janela `WINDOW` e seed; para reproducibilidade, teste `TF_ENABLE_ONEDNN_OPTS=0`.
- **Gráficos com rolagem/overflows** → ver CSS de `.canvas-container`/`#hero-canvas` e cards.
