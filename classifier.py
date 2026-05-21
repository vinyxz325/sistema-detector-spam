"""
classifier.py — Núcleo NLP com TensorFlow/Keras
Arquitetura: Embedding + LSTM Bidirecional + Dense
"""

import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# ─── Dataset de treinamento ────────────────────────────────────────────────────
# Formato: (texto, label) onde 1 = SPAM, 0 = LEGÍTIMO
# Exemplos representativos de DMs para influenciadores

TRAINING_DATA = [
    # ── SPAM: Golpes financeiros ──────────────────────────────────────────────
    ("Ganhe R$5000 por semana trabalhando de casa! Cadastre-se agora: bit.ly/ganhe5k", 1),
    ("Oi! Você foi selecionado para nosso programa de renda extra. Clique aqui!", 1),
    ("Invista apenas R$100 e receba R$3000 em 48 horas. Método 100% garantido!", 1),
    ("Olá influencer! Temos uma proposta incrível de R$10.000 mensais para você!", 1),
    ("Dinheiro rápido e fácil! Acesse o link e comece hoje mesmo a ganhar muito!", 1),
    ("Promoção exclusiva: multiplique seu dinheiro em 3 dias! Link na bio: scam.com", 1),
    ("Sistema automático de renda passiva! Já pagou mais de R$50 milhões. Acesse!", 1),
    ("Atenção! Você tem um bônus de R$500 esperando. Clique para resgatar agora!", 1),
    ("Trabalhe apenas 2 horas por dia e ganhe mais de R$8000 mensais. Saiba como!", 1),
    ("Oportunidade única! Seja nosso parceiro e fature alto sem sair de casa!", 1),
    ("Rendimento diário garantido de 5%! Invista agora antes que encerre as vagas", 1),
    ("Olá! Vi seu perfil e tenho uma oferta de renda passiva exclusiva para você", 1),
    ("Plataforma de investimento paga até 300% em 7 dias. Corra e cadastre-se!", 1),
    ("Vaga home office urgente! Ganhe R$2500 semanais respondendo mensagens.", 1),
    ("Cadastre-se agora e ganhe R$200 de bônus na primeira semana garantido!", 1),

    # ── SPAM: Phishing e links maliciosos ─────────────────────────────────────
    ("Seu perfil foi selecionado para verificação. Acesse: verify-insta.net agora!", 1),
    ("Instagram detectou atividade suspeita. Confirme seu login: insta-secure.tk", 1),
    ("Você ganhou um iPhone 15! Clique aqui para resgatar: premio-sorteio.xyz", 1),
    ("URGENTE: Sua conta será desativada em 24h. Verifique: account-protect.net", 1),
    ("Clique no link para receber seu pagamento de patrocínio: bit.ly/pagamento99", 1),
    ("Verificação pendente! Acesse http://login-seguro.club para não perder acesso", 1),
    ("Você foi marcado em um sorteio! Resgate: free-prize-2024.com/influencer", 1),
    ("Atenção: sua conta foi comprometida! Acesse imediatamente: reset-senha.tk", 1),
    ("Parceria disponível! Veja os detalhes aqui: collab-pro.xyz/offer/influencer", 1),
    ("Receba R$1000 de patrocínio toda semana. Cadastro: sponsor-pay.net/register", 1),
    ("Hacker alerta! Alguém tentou acessar sua conta. Proteja agora: safeinsta.ml", 1),
    ("SEU PRÊMIO EXPIROU EM 2H. Resgate agora: ganhador-oficial.com/premio", 1),
    ("Atualização obrigatória de segurança. Acesse: instagram-update.xyz urgente!", 1),
    ("Parabéns! Você foi sorteado. Para receber acesse: sorteio-verdadeiro.net", 1),
    ("Marca quer te patrocinar! Contrato disponível em: brand-deal.xyz/contratar", 1),

    # ── SPAM: Compra de seguidores/engajamento ────────────────────────────────
    ("Compre 10.000 seguidores reais por apenas R$49! Entrega em 24h garantida!", 1),
    ("Aumente seu engajamento agora! 5000 likes reais por R$29. Peça já!", 1),
    ("Seguidores brasileiros reais! 1000 por R$15. Resultados em 1 hora!", 1),
    ("Impulsione seu perfil! Compre visualizações, curtidas e comentários baratos", 1),
    ("Pacote especial para influencers: 50k seguidores + 10k likes por R$199!", 1),
    ("Venda de contas verificadas com seguidores! Preço especial hoje apenas!", 1),
    ("Boost seu Instagram agora! Seguidores, curtidas, comentários reais e baratos", 1),
    ("Serviço de crescimento orgânico: 100k seguidores em 30 dias. Confira preço!", 1),

    # ── SPAM: Promoções e produtos falsos ─────────────────────────────────────
    ("Produtos emagrecedores com desconto especial! Emagreça 15kg em 30 dias!", 1),
    ("Remédio milagroso para emagrecer! Aprovado pela ANVISA. Peça já o seu!", 1),
    ("Criptomoeda vai explodir essa semana! Compre agora antes que suba 1000%!", 1),
    ("Aprenda a ganhar dinheiro com Bitcoin! Curso grátis: cripto-milionario.com", 1),
    ("Iphone 15 Pro Max por R$800! Compre agora antes do estoque acabar!", 1),
    ("Louis Vuitton, Gucci e Rolex com 90% OFF! Loja oficial: luxury-deal.net", 1),
    ("Promoção relâmpago! Produtos importados com frete grátis. Acesse agora!", 1),
    ("Ganhe dinheiro com meu método secreto forex! Resultados comprovados diários", 1),

    # ── SPAM: Assédio e conteúdo adulto ──────────────────────────────────────
    ("Oi gata! Tenho fotos suas que vou publicar se não me responder. Fala comigo", 1),
    ("Vi suas fotos e fiquei com vontade. Me manda seu número que te pago bem", 1),
    ("Cuidado: tenho acesso à sua conta. Pague R$500 em Bitcoin ou publico tudo", 1),
    ("Conteúdo adulto exclusivo! Acesse meu canal privado: onlyfake-content.com", 1),

    # ── LEGÍTIMO: Fãs genuínos ────────────────────────────────────────────────
    ("Amei o vídeo de hoje! Você me inspira demais, continua fazendo esse conteúdo!", 0),
    ("Oi! Assisto seus vídeos toda semana, você é incrível! Obrigada pelo conteúdo", 0),
    ("Cara, aquele tutorial de ontem salvou minha vida! Muito obrigado mesmo!", 0),
    ("Sou fã há 3 anos! Seu trabalho mudou minha vida. Amo demais você!", 0),
    ("Minha filha te ama! Ela quer ser influencer igual a você quando crescer 😄", 0),
    ("Vi seu post sobre ansiedade e chorei muito, me identifiquei 100%. Obrigada!", 0),
    ("Você poderia fazer um vídeo sobre finanças pessoais? Adoraria esse conteúdo!", 0),
    ("Descobri seu canal essa semana e já assisti todos os vídeos! Incrível demais!", 0),
    ("Oi! Onde você comprou essa roupa do último vídeo? Adorei o look!", 0),
    ("Seu conteúdo de viagem me motivou a finalmente tirar as férias que precisava!", 0),
    ("Gostaria de saber mais sobre sua rotina de skincare. Sua pele é maravilhosa!", 0),
    ("Você vai vir para o Rio? Adoraria te encontrar num meet and greet!", 0),
    ("Compartilhei seu vídeo com toda a minha família. Todos amaram muito!", 0),
    ("Qual câmera você usa? Quero começar um canal igual ao seu dia de hoje!", 0),
    ("Seu jeito de falar sobre sustentabilidade mudou minha perspectiva. Obrigada!", 0),

    # ── LEGÍTIMO: Propostas reais de parceria ─────────────────────────────────
    ("Olá! Sou da equipe de marketing da Natura. Gostaríamos de uma parceria.", 0),
    ("Boa tarde! Meu nome é Camila, trabalho na agência X. Posso te chamar pelo email?", 0),
    ("Representante da marca iFood aqui. Temos interesse numa collab. Aceita?", 0),
    ("Olá, somos da Samsung Brasil. Adoramos seu conteúdo tech. Podemos conversar?", 0),
    ("Equipe Reserva aqui! Vimos seu estilo e queremos propor uma parceria.", 0),
    ("Boa tarde! Marca de moda sustentável gostaria de enviar peças para review.", 0),
    ("Oi! Trabalho com influencer marketing na Havaianas. Posso enviar proposta?", 0),
    ("Gestor de marca aqui. Adoramos seu engajamento. Posso mandar briefing por email?", 0),
    ("Olá! Somos a plataforma MagaluPartners. Temos proposta de afiliados para você.", 0),
    ("Revista Vogue Brasil! Queremos te convidar para nossa próxima editorial.", 0),

    # ── LEGÍTIMO: Networking profissional ─────────────────────────────────────
    ("Oi! Também sou criador de conteúdo e adoraria fazer uma collab contigo!", 0),
    ("Sou fotógrafo e adoraria fazer um ensaio em troca de divulgação no seu perfil", 0),
    ("Olá! Faço podcasts e você seria uma convidada incrível. Posso te convidar?", 0),
    ("Designer gráfico aqui! Vi seu trabalho e gostaria de trocar experiências.", 0),
    ("Oi! Somos um grupo de criadores de SP. Vamos marcar uma troca de experiências?", 0),
    ("Produtora de eventos aqui. Quer ser embaixadora do nosso festival de verão?", 0),
    ("Olá! Escrevo para blog de marketing digital. Posso te entrevistar?", 0),
    ("Equipe do podcast Millenials aqui. Você quer ser nossa convidada especial?", 0),

    # ── LEGÍTIMO: Perguntas e comentários cotidianos ───────────────────────────
    ("Qual a receita daquele bolo que você fez no story? Me manda por favor!", 0),
    ("Você recomenda aquele curso de inglês que patrocinou? Vale a pena mesmo?", 0),
    ("Oi! Vi que você foi pra Portugal. Qual hotel ficou? Vou viajar ano que vem!", 0),
    ("Aquele produto de cabelo que você usou semana passada, qual era a marca?", 0),
    ("Você pode indicar um nutricionista? Vi que você emagreceu de forma saudável!", 0),
    ("Quanto tempo você levou para chegar a 100k seguidores? Pergunta de fã!", 0),
    ("Quando sai o próximo vídeo? Estou ansiosa esperando! Adoro seu conteúdo!", 0),
    ("Oi! Você vai ao festival Lollapalooza? Queria te ver lá!", 0),
    ("Você faz consultorias de marketing? Tenho um negócio pequeno e adorei suas dicas", 0),
    ("Adorei sua colaboração com a marca X! Os produtos são incríveis mesmo!", 0),
]


# ─── Sinais de spam para análise explícita ────────────────────────────────────
SPAM_SIGNALS = {
    "link suspeito": r"bit\.ly|tinyurl|t\.co\/|goo\.gl|short\.link|ow\.ly|"
                     r"\.tk|\.ml|\.xyz|\.club|\.net\/|\.online|\.site",
    "dinheiro fácil": r"ganhe?\s+r\$|rend[ao]\s+passiv[ao]|fatur[ae]\s+alto|"
                      r"dinheiro\s+r[aá]pid|renda\s+extr[ao]|lucro\s+garant",
    "promessa exagerada": r"\d+\s*%\s*(ao\s*dia|diári|de\s*lucro)|"
                          r"(tripli|dupli|multi)qu[ea]|garantid[ao]|sem\s+risco",
    "urgência falsa": r"urgente|expira\s+em|últimas?\s+vaga|encerra\s+hoje|"
                      r"agora\s+mesmo|corra\s+e|não\s+perca",
    "seguidores falsos": r"compre?\s+seguidores?|seguidores?\s+reais?|"
                         r"\d+k?\s+likes?|boost\s+seu|impulsione\s+seu",
    "phishing": r"verifi(que|car|cação)|conta\s+ser[aá]\s+desativ|"
                r"atividade\s+suspeita|acesse\s+agora|clique\s+aqui",
    "ameaça/chantagem": r"vou\s+public(ar|o)|tenho\s+fotos?|tenho\s+acesso|"
                         r"pague?\s+ou|bitcoin|criptomoeda",
    "produto milagroso": r"emagrec[ea]|(15|20|30)\s*kg\s+em|milagroso|"
                         r"aprovado\s+pela?\s+anvisa",
}


class SpamClassifier:
    """
    Classificador de spam para DMs usando TensorFlow.

    Arquitetura:
        Input (texto) → Tokenizer → Embedding (64d) →
        Bidirecional LSTM (64u) → Dropout → Dense(32, relu) →
        Dense(1, sigmoid) → P(spam)
    """

    MAX_VOCAB  = 5_000   # Tamanho do vocabulário
    MAX_LEN    = 150     # Comprimento máximo da sequência
    EMBED_DIM  = 64      # Dimensão dos embeddings
    LSTM_UNITS = 64      # Unidades no LSTM

    def __init__(self):
        self.tokenizer   = None
        self.model       = None
        self._accuracy   = 0.0
        self.vocab_size  = 0
        self.train_size  = 0

    # ── Pré-processamento de texto ─────────────────────────────────────────────
    @staticmethod
    def preprocess(text: str) -> str:
        text = text.lower()
        # Preserva padrões importantes como URLs e valores monetários
        text = re.sub(r'r\$\s*(\d+)', r'dinheiro \1 reais', text)
        text = re.sub(r'https?://\S+', 'link_externo', text)
        text = re.sub(r'\S+\.(tk|ml|xyz|club|net|com|online)\S*', 'link_suspeito', text)
        text = re.sub(r'bit\.ly\S*|tinyurl\S*|goo\.gl\S*', 'link_encurtado', text)
        text = re.sub(r'\d+%', 'percentual', text)
        text = re.sub(r'[^\w\sáàâãéêíóôõúç_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── Construção e treinamento do modelo ────────────────────────────────────
    def build_and_train(self):
        texts  = [self.preprocess(t) for t, _ in TRAINING_DATA]
        labels = np.array([l for _, l in TRAINING_DATA], dtype=np.float32)

        self.train_size = len(texts)

        # Tokenização
        self.tokenizer = Tokenizer(
            num_words=self.MAX_VOCAB,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',
            lower=True
        )
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = min(len(self.tokenizer.word_index) + 1, self.MAX_VOCAB)

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded    = pad_sequences(sequences, maxlen=self.MAX_LEN,
                                  padding='post', truncating='post')

        # Shuffle e split (80/20)
        indices = np.random.permutation(len(padded))
        split   = int(0.8 * len(indices))
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, y_train = padded[train_idx], labels[train_idx]
        X_val,   y_val   = padded[val_idx],   labels[val_idx]

        # Arquitetura LSTM Bidirecional
        self.model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.EMBED_DIM,
                input_length=self.MAX_LEN,
                name="embedding"
            ),
            Bidirectional(
                LSTM(self.LSTM_UNITS, return_sequences=True, dropout=0.2),
                name="bilstm"
            ),
            GlobalMaxPooling1D(name="pooling"),
            Dense(32, activation='relu', name="dense_1"),
            Dropout(0.4, name="dropout"),
            Dense(1, activation='sigmoid', name="output")
        ], name="spam_classifier")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Treinamento com early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=0
        )

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

        # Avalia no conjunto de validação
        _, acc = self.model.evaluate(X_val, y_val, verbose=0)
        self._accuracy = float(acc)

    # ── Predição ──────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        """
        Retorna dict com:
            is_spam    : bool
            spam_prob  : float [0, 1]
            signals    : list[str]  — sinais específicos detectados no texto
        """
        clean_text = self.preprocess(text)
        seq    = self.tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=self.MAX_LEN, padding='post')
        prob   = float(self.model.predict(padded, verbose=0)[0][0])

        # Detecta sinais explícitos de spam no texto original
        signals = self._detect_signals(text)

        # Ajuste baseado em sinais explícitos detectados
        if signals and prob < 0.6:
            prob = min(prob + 0.15 * len(signals), 0.95)

        return {
            "is_spam":   prob >= 0.5,
            "spam_prob": prob,
            "signals":   signals
        }

    def _detect_signals(self, text: str) -> list:
        """Detecta padrões suspeitos explícitos no texto."""
        found = []
        lower = text.lower()
        for name, pattern in SPAM_SIGNALS.items():
            if re.search(pattern, lower, re.IGNORECASE):
                found.append(name)
        return found

    def get_accuracy(self) -> float:
        return self._accuracy