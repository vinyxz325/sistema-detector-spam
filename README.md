# 🛡️ DM Shield — Detector de Spam para Influenciadores

App Streamlit com TensorFlow/Keras para classificar mensagens privadas (DMs) como spam ou legítimas.

## Arquitetura do Modelo

```
Texto → Tokenizer → Embedding (64d) → BiLSTM (64u) → GlobalMaxPool → Dense(32) → Sigmoid
```

## Instalação

```bash
pip install -r requirements.txt
```

## Executar

```bash
streamlit run app.py
```

## Estrutura

```
spam_detector/
├── app.py           # Interface Streamlit
├── classifier.py    # Modelo TensorFlow + dataset de treino
├── requirements.txt
└── README.md
```

## O que é classificado como SPAM

| Categoria               | Exemplos                                           |
|-------------------------|----------------------------------------------------|
| Golpes financeiros      | "Ganhe R$5000 por semana", renda passiva, pirâmide |
| Phishing / links        | Links encurtados, páginas falsas de verificação    |
| Compra de seguidores    | Venda de likes, followers, engajamento falso       |
| Produtos miraculosos    | Emagrecimento impossível, crypto esquemas          |
| Ameaças / chantagem     | Extorsão, conteúdo adulto não solicitado           |

## O que é classificado como LEGÍTIMO

| Categoria              | Exemplos                                            |
|------------------------|-----------------------------------------------------|
| Fãs genuínos           | Elogios, perguntas, agradecimentos                  |
| Parcerias reais        | Marcas conhecidas com proposta profissional         |
| Networking             | Outros criadores, fotógrafos, produtoras            |
| Perguntas cotidianas   | Produto usado, destino de viagem, rotina            |

## Processamento Assíncrono

A classificação roda em **thread separada** via `threading.Thread`, garantindo que a UI Streamlit não trave durante a inferência do modelo.
