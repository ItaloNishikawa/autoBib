"""
Módulo de Inteligência Artificial (Groq).

Responsabilidade: isolar toda a comunicação com a API do Groq.
Usado como provedor alternativo quando a cota do Google Gemini é esgotada,
ou como provedor principal caso o usuário prefira.

Prompts e parser JSON são compartilhados com ai_gemini para evitar duplicação.
"""

import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Reutiliza os prompts e o parser de JSON do módulo Gemini
from modules.ai_gemini import _build_prompt, _parse_json_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cadeia de modelos Groq tentados em ordem (fallback interno)
# ---------------------------------------------------------------------------

_GROQ_MODELS: list[str] = [
    "openai/gpt-oss-120b",      # modelo principal (via roteamento Groq)
    "llama-3.3-70b-versatile",  # melhor modelo open-source na plataforma
    "llama-3.1-8b-instant",     # mais leve, último recurso
]

# ---------------------------------------------------------------------------
# Inicialização lazy do cliente
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    """Retorna o cliente Groq, inicializando-o na primeira chamada."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY não encontrada. Defina a variável no arquivo .env."
        )
    return Groq(api_key=api_key)


def _is_quota_error(exc: Exception) -> bool:
    """Retorna True se a exceção indicar limite de cota (HTTP 429)."""
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "quota" in msg


def _generate_with_fallback(client: Groq, prompt: str, on_step=None) -> str:
    """Tenta gerar conteúdo iterando pela cadeia de modelos em ``_GROQ_MODELS``.

    Quando a API retorna 429 (rate limit), aguarda 60 s e retenta o mesmo
    modelo uma vez antes de avançar para o próximo da lista.

    Args:
        on_step: Callback opcional ``(mensagem, percentual)`` — percentual -2 indica
                 mensagem informativa de espera por cota.

    Returns:
        Texto bruto retornado pelo modelo que teve sucesso.
    """
    _RETRY_WAIT = 60  # segundos de espera ao receber 429

    last_exc: Exception = RuntimeError("Nenhum modelo Groq disponível em _GROQ_MODELS.")

    for model in _GROQ_MODELS:
        for attempt in (1, 2):  # tenta o mesmo modelo até 2 vezes antes de avançar
            try:
                logger.debug("Groq: tentando modelo '%s' (tentativa %d)...", model, attempt)
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_completion_tokens=8192,
                    top_p=1,
                    stream=True,
                    stop=None,
                )

                result = ""
                for chunk in stream:
                    result += chunk.choices[0].delta.content or ""

                logger.debug("Groq: modelo '%s' respondeu com sucesso.", model)
                return result

            except Exception as exc:  # noqa: BLE001
                if _is_quota_error(exc):
                    if attempt == 1:
                        wait_msg = (
                            f"⏳ Rate limit atingido (modelo {model}). "
                            f"Aguardando {_RETRY_WAIT}s para a próxima requisição..."
                        )
                        logger.warning(
                            "Groq: modelo '%s' atingiu o rate limit. Aguardando %ds...",
                            model, _RETRY_WAIT,
                        )
                        if on_step:
                            on_step(wait_msg, -2)
                        time.sleep(_RETRY_WAIT)
                        continue  # retenta o mesmo modelo
                    else:
                        logger.warning(
                            "Groq: modelo '%s' ainda com rate limit após espera. Tentando próximo...",
                            model,
                        )
                        last_exc = exc
                        break  # avança para o próximo modelo
                raise  # outros erros são relançados imediatamente

    raise last_exc


# ---------------------------------------------------------------------------
# Funções públicas do módulo (mesma interface que ai_gemini)
# ---------------------------------------------------------------------------

def generate_query_and_justification(
    theme: str, database: str
) -> tuple[str, str] | tuple[None, None]:
    """Gera uma string de busca otimizada e sua justificativa técnica via Groq.

    Args:
        theme: Tema da pesquisa (ex: ``"Post-Quantum Cryptography"``).
        database: Base de dados alvo. Valores aceitos: ``"Scopus"``, ``"IEEE"``, ``"ACM"``.

    Returns:
        Tupla ``(query, justification)`` com as strings geradas pela IA, ou
        ``(None, None)`` em caso de falha.
    """
    logger.info("[Groq] Gerando query para '%s' (base: %s)...", theme, database)

    prompt = _build_prompt(theme, database)

    try:
        client  = _get_client()
        text    = _generate_with_fallback(client, prompt)
        ai_data = _parse_json_response(text)
        logger.info("[Groq] Query gerada com sucesso.")
        return ai_data["query"], ai_data["justification"]

    except (ValueError, KeyError) as exc:
        logger.error("[Groq] Erro ao processar resposta da IA: %s", exc)
    except Exception as exc:
        logger.error("[Groq] Erro ao comunicar com o Groq: %s", exc)

    return None, None


def analyze_abstracts(df: pd.DataFrame, on_step=None) -> pd.DataFrame:
    """Envia os abstracts para a IA Groq gerar a análise crítica estruturada.

    Para cada artigo com abstract válido, consulta o Groq e armazena
    o resultado formatado em 5 tópicos na coluna ``observations``.

    Args:
        df:      DataFrame (preferencialmente já deduplicado e enriquecido).
        on_step: Callback opcional ``(mensagem, percentual)`` para progresso na UI.
                 Percentual -1 = aviso de erro, -2 = info de espera por cota.

    Returns:
        DataFrame com a coluna ``observations`` adicionada.
    """
    if df.empty:
        logger.warning("[Groq] DataFrame vazio. Pulando análise de abstracts.")
        return df

    df     = df.copy()
    df["observations"] = None

    client = _get_client()
    total  = int(df["abstract"].notna().sum())
    logger.info("[Groq] Iniciando leitura crítica com IA para %d artigos...", total)

    done = 0
    for idx, row in df.iterrows():
        abstract = row.get("abstract", "")
        title    = row.get("title", "Artigo sem título")

        if not abstract or len(str(abstract)) < 50:
            df.at[idx, "observations"] = "Resumo ausente ou muito curto para análise."
            continue

        done += 1
        pct_start  = int((done - 1) / max(total, 1) * 100)
        short_title = str(title)[:60]

        logger.info("[Groq] Analisando: '%s'...", short_title)
        if on_step:
            on_step(f"🔍 ({done}/{total}) Analisando: '{short_title}'...", pct_start)

        prompt = f"""
        Leia o abstract deste artigo científico.
        Eu preciso que você extraia uma análise crítica estruturada.

        Retorne APENAS um JSON estrito com uma única chave chamada "observations".
        O valor dessa chave deve ser um texto contendo exatamente estes 5 tópicos (use quebra de linha \\n):
        - Contribuição principal: ...
        - Metodologia usada: ...
        - Limitações identificadas: ...
        - Potencial de aplicação: ...
        - Relevância para pesquisa futura: ...

        NÃO inclua saudações, explicações fora do JSON ou blocos de formatação markdown.

        Abstract: {abstract}
        """

        try:
            text    = _generate_with_fallback(client, prompt, on_step)
            ai_data = _parse_json_response(text)
            df.at[idx, "observations"] = ai_data.get(
                "observations", "Erro na formatação da resposta."
            )
            pct_done = int(done / max(total, 1) * 100)
            if on_step:
                on_step(f"✅ ({done}/{total}) Concluído: '{short_title}'", pct_done)
        except Exception as exc:
            logger.warning("[Groq] Erro ao analisar abstract (índice %d): %s", idx, exc)
            df.at[idx, "observations"] = "Erro na análise da IA."
            if on_step:
                on_step(
                    f"⚠️ ({done}/{total}) Erro ao analisar '{str(title)[:40]}': {type(exc).__name__}",
                    -1,
                )

        # O espaçamento entre requisições é gerenciado automaticamente
        # por _generate_with_fallback ao receber 429.

    logger.info("[Groq] Leitura crítica finalizada.")
    return df
