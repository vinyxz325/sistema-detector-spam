import streamlit as st
import time
import threading
import queue
from classifier import SpamClassifier
 
# ─── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="DM Shield — Detector de Spam",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)
 
# ─── CSS Customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
 
/* Reset e base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
 
/* Background escuro com textura */
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(255,60,0,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(0,200,255,0.05) 0%, transparent 40%);
}
 
/* Header principal */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
}
.main-header h1 {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ff3c00 0%, #ff8c00 50%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: #666;
    font-size: 1rem;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
}
 
/* Caixa de texto */
.stTextArea textarea {
    background: #111118 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: #ff3c00 !important;
    box-shadow: 0 0 0 3px rgba(255,60,0,0.12) !important;
}
 
/* Botão principal */
.stButton > button {
    background: linear-gradient(135deg, #ff3c00, #ff6a00) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,60,0,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
 
/* Resultado - SPAM */
.result-spam {
    background: linear-gradient(135deg, rgba(255,30,30,0.12), rgba(255,60,0,0.08));
    border: 1.5px solid rgba(255,60,60,0.4);
    border-left: 4px solid #ff3030;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin: 1.2rem 0;
    animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.result-spam .verdict {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ff4040;
    letter-spacing: -0.02em;
}
.result-spam .detail {
    color: #cc6060;
    font-size: 0.88rem;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
 
/* Resultado - LEGÍTIMO */
.result-clean {
    background: linear-gradient(135deg, rgba(30,255,120,0.08), rgba(0,200,100,0.05));
    border: 1.5px solid rgba(30,200,100,0.3);
    border-left: 4px solid #1ec86e;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin: 1.2rem 0;
    animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.result-clean .verdict {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1ec86e;
    letter-spacing: -0.02em;
}
.result-clean .detail {
    color: #5aaa80;
    font-size: 0.88rem;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
 
/* Barra de confiança */
.confidence-bar-wrap {
    margin-top: 1rem;
}
.confidence-label {
    font-size: 0.78rem;
    color: #666;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.4rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.confidence-bar-bg {
    background: #1a1a2a;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.confidence-bar-fill-spam {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #ff3030, #ff6a00);
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.confidence-bar-fill-clean {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1ec86e, #00d4aa);
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}
 
/* Sinais detectados */
.signals-box {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.signals-title {
    font-size: 0.75rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.6rem;
}
.signal-tag {
    display: inline-block;
    background: rgba(255,60,0,0.12);
    color: #ff7040;
    border: 1px solid rgba(255,60,0,0.2);
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
.signal-tag-neutral {
    background: rgba(100,100,150,0.12);
    color: #7878aa;
    border-color: rgba(100,100,150,0.2);
}
 
/* Estatísticas do modelo */
.stats-row {
    display: flex;
    gap: 0.8rem;
    margin: 1.5rem 0 0.5rem;
}
.stat-card {
    flex: 1;
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.7rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
 
/* Histórico */
.history-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: #111118;
    border: 1px solid #1e1e2e;
    font-size: 0.82rem;
    color: #888;
    font-family: 'JetBrains Mono', monospace;
}
.dot-spam { color: #ff4040; }
.dot-clean { color: #1ec86e; }
 
/* Divisor */
.divider {
    border: none;
    border-top: 1px solid #1a1a2a;
    margin: 1.5rem 0;
}
 
/* Animação */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
 
/* Spinner customizado */
.loading-wrap {
    text-align: center;
    padding: 1.5rem;
    color: #555;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}
 
/* Ocultar elementos padrão do Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 720px; }
</style>
""", unsafe_allow_html=True)
 
# ─── Inicialização do classificador (cache para não recarregar) ─────────────────
@st.cache_resource(show_spinner=False)
def load_classifier():
    clf = SpamClassifier()
    clf.build_and_train()
    return clf
 
# ─── Estado da sessão ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "loading" not in st.session_state:
    st.session_state.loading = False
 
# ─── Layout principal ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ DM Shield</h1>
    <p>DETECTOR DE SPAM PARA INFLUENCIADORES · POWERED BY TENSORFLOW</p>
</div>
""", unsafe_allow_html=True)
 
# Carrega o modelo
with st.spinner("Carregando modelo TensorFlow..."):
    classifier = load_classifier()
 
# ─── Estatísticas do modelo ────────────────────────────────────────────────────
acc = classifier.get_accuracy()
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">{acc:.0%}</div>
        <div class="stat-label">Acurácia</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{classifier.vocab_size:,}</div>
        <div class="stat-label">Vocabulário</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{classifier.train_size:,}</div>
        <div class="stat-label">Exemplos</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">LSTM</div>
        <div class="stat-label">Arquitetura</div>
    </div>
</div>
""", unsafe_allow_html=True)
 
st.markdown('<hr class="divider">', unsafe_allow_html=True)
 
# ─── Input de mensagem ─────────────────────────────────────────────────────────
dm_text = st.text_area(
    "Cole aqui a mensagem recebida:",
    placeholder="Ex: 'Oi! Ganhe R$5000 reais por semana trabalhando de casa! Clique no link: bit.ly/oferta123 🤑🤑'",
    height=130,
    key="dm_input",
    label_visibility="visible",
)
 
col1, col2 = st.columns([3, 1])
with col1:
    analisar = st.button("🔍 ANALISAR MENSAGEM", use_container_width=True)
with col2:
    limpar = st.button("Limpar", use_container_width=True)
 
if limpar:
    st.session_state.result = None
    st.rerun()
 
# ─── Processamento assíncrono com queue ───────────────────────────────────────
result_queue = queue.Queue()
 
def classify_async(text, q):
    """Roda a classificação em thread separada para não travar a UI."""
    try:
        result = classifier.predict(text)
        q.put(("ok", result))
    except Exception as e:
        q.put(("error", str(e)))
 
if analisar and dm_text.strip():
    # Exibe indicador de carregamento imediatamente
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        '<div class="loading-wrap">⚡ ANALISANDO MENSAGEM...</div>',
        unsafe_allow_html=True
    )
 
    # Executa em thread separada
    t = threading.Thread(target=classify_async, args=(dm_text.strip(), result_queue))
    t.start()
    t.join(timeout=15)  # timeout de segurança
 
    loading_placeholder.empty()
 
    if not result_queue.empty():
        status, payload = result_queue.get()
        if status == "ok":
            st.session_state.result = payload
            # Adiciona ao histórico (máx 10 itens)
            preview = dm_text.strip()[:55] + ("…" if len(dm_text.strip()) > 55 else "")
            st.session_state.history.insert(0, {
                "preview": preview,
                "is_spam": payload["is_spam"],
                "confidence": payload["spam_prob"]
            })
            st.session_state.history = st.session_state.history[:10]
        else:
            st.error(f"Erro na análise: {payload}")
    else:
        st.warning("Tempo de análise excedido. Tente novamente.")
 
elif analisar and not dm_text.strip():
    st.warning("Cole uma mensagem para analisar.")
 
# ─── Exibição do resultado ────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    pct = int(r["spam_prob"] * 100)
    not_pct = 100 - pct
 
    if r["is_spam"]:
        bar_class = "confidence-bar-fill-spam"
        signals_html = "".join(
            f'<span class="signal-tag">⚠ {s}</span>' for s in r["signals"]
        )
        st.markdown(f"""
        <div class="result-spam">
            <div class="verdict">🚨 SPAM DETECTADO</div>
            <div class="detail">Confiança: {pct}% · Classificado como mensagem maliciosa</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">Probabilidade de Spam</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill-spam" style="width:{pct}%"></div>
                </div>
            </div>
            <div class="signals-box">
                <div class="signals-title">⚡ Sinais detectados</div>
                {signals_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        signals_html = "".join(
            f'<span class="signal-tag-neutral">{s}</span>' for s in r["signals"]
        ) if r["signals"] else '<span class="signal-tag-neutral">nenhum sinal suspeito</span>'
        st.markdown(f"""
        <div class="result-clean">
            <div class="verdict">✅ MENSAGEM LEGÍTIMA</div>
            <div class="detail">Confiança: {not_pct}% · Nenhuma ameaça detectada</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">Probabilidade Legítima</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill-clean" style="width:{not_pct}%"></div>
                </div>
            </div>
            <div class="signals-box">
                <div class="signals-title">📋 Análise</div>
                {signals_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
 
# ─── Histórico de análises ─────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#444;font-size:0.75rem;text-transform:uppercase;'
        'letter-spacing:0.1em;font-family:JetBrains Mono,monospace;margin-bottom:0.6rem">'
        '📋 Histórico desta sessão</p>',
        unsafe_allow_html=True
    )
    for item in st.session_state.history:
        icon = "🔴" if item["is_spam"] else "🟢"
        label = "SPAM" if item["is_spam"] else "OK"
        conf = int(item["confidence"] * 100) if item["is_spam"] else int((1 - item["confidence"]) * 100)
        st.markdown(f"""
        <div class="history-item">
            <span>{icon}</span>
            <span style="color:{'#ff4040' if item['is_spam'] else '#1ec86e'};
                          font-weight:600;min-width:42px">{label}</span>
            <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{item['preview']}</span>
            <span style="color:#444">{conf}%</span>
        </div>
        """, unsafe_allow_html=True)