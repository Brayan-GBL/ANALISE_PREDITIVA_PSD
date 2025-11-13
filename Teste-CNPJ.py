import re
import json
import sqlite3
from datetime import datetime, timedelta

import requests
import streamlit as st


# ===========================
# CONFIG
# ===========================

API_URL = "https://publica.cnpj.ws/cnpj/{cnpj}"
DB_PATH = "cnpj_cache.db"
CACHE_TTL_DAYS = 60  # ~2 meses


# ===========================
# FUNÇÕES DE UTILIDADE
# ===========================

def limpar_cnpj(cnpj: str) -> str:
    """Remove tudo que não for dígito."""
    return re.sub(r"\D", "", cnpj or "")


def formatar_cnpj(cnpj: str) -> str:
    cnpj = limpar_cnpj(cnpj)
    if len(cnpj) != 14:
        return cnpj
    return f"{cnpj[0:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:14]}"


def init_db():
    """Cria o banco e a tabela de cache se não existirem."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cnpj_cache (
            cnpj TEXT PRIMARY KEY,
            razao_social TEXT,
            ie_principal TEXT,
            ufs_ies TEXT,              -- JSON com lista de {uf, ie, ativo}
            atualizado_em_api TEXT,    -- data/hora do dado na API (campo atualizado_em)
            atualizado_em_cache TEXT   -- data/hora que salvamos no cache
        )
        """
    )
    conn.commit()
    return conn


def carregar_cache(cnpj: str):
    conn = init_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT razao_social, ie_principal, ufs_ies, atualizado_em_api, atualizado_em_cache "
        "FROM cnpj_cache WHERE cnpj = ?",
        (cnpj,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    razao_social, ie_principal, ufs_ies_json, atualizado_api, atualizado_cache = row

    try:
        atualizado_cache_dt = datetime.fromisoformat(atualizado_cache)
    except Exception:
        atualizado_cache_dt = None

    return {
        "cnpj": cnpj,
        "razao_social": razao_social,
        "ie_principal": ie_principal,
        "ufs_ies": json.loads(ufs_ies_json),
        "atualizado_em_api": atualizado_api,
        "atualizado_em_cache": atualizado_cache_dt,
    }


def salvar_cache(cnpj: str, dados: dict):
    conn = init_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO cnpj_cache
        (cnpj, razao_social, ie_principal, ufs_ies, atualizado_em_api, atualizado_em_cache)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            cnpj,
            dados.get("razao_social"),
            dados.get("ie_principal"),
            json.dumps(dados.get("ufs_ies", []), ensure_ascii=False),
            dados.get("atualizado_em_api"),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def cache_esta_valido(registro: dict) -> bool:
    """Verifica se o cache tem menos de CACHE_TTL_DAYS dias."""
    if not registro or not registro.get("atualizado_em_cache"):
        return False
    limite = datetime.now() - timedelta(days=CACHE_TTL_DAYS)
    return registro["atualizado_em_cache"] >= limite


# ===========================
# CHAMADA À API
# ===========================

def consultar_api_cnpjws(cnpj: str) -> dict:
    url = API_URL.format(cnpj=cnpj)
    resp = requests.get(url, timeout=20)

    if resp.status_code == 404:
        raise ValueError("CNPJ não encontrado na API pública (404).")

    if resp.status_code == 429:
        raise RuntimeError("Limite de requisições atingido (429). Aguarde um pouco e tente novamente.")

    if resp.status_code != 200:
        raise RuntimeError(f"Erro HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()

    razao_social = data.get("razao_social")
    atualizado_em_api = data.get("atualizado_em")

    est = data.get("estabelecimento", {}) or {}
    estado_info = est.get("estado") or {}
    uf_principal = estado_info.get("sigla")

    ies = est.get("inscricoes_estaduais") or []

    # Monta a lista de IEs com UF
    lista_ies = []
    for item in ies:
        est_uf = (item.get("estado") or {}).get("sigla")
        lista_ies.append(
            {
                "uf": est_uf,
                "inscricao_estadual": item.get("inscricao_estadual"),
                "ativo": bool(item.get("ativo")),
                "atualizado_em": item.get("atualizado_em"),
            }
        )

    # Escolhe IE principal:
    ie_principal = None

    # 1) tenta pegar IE ativa da UF principal
    if uf_principal:
        for item in lista_ies:
            if item["uf"] == uf_principal and item["ativo"] and item["inscricao_estadual"]:
                ie_principal = item["inscricao_estadual"]
                break

    # 2) se não achar, pega qualquer IE ativa
    if not ie_principal:
        for item in lista_ies:
            if item["ativo"] and item["inscricao_estadual"]:
                ie_principal = item["inscricao_estadual"]
                break

    # 3) se ainda não achar, pega a primeira IE
    if not ie_principal and lista_ies:
        ie_principal = lista_ies[0]["inscricao_estadual"]

    return {
        "cnpj": cnpj,
        "razao_social": razao_social,
        "ie_principal": ie_principal,
        "ufs_ies": lista_ies,
        "atualizado_em_api": atualizado_em_api,
    }


def obter_dados_cnpj(cnpj: str):
    """
    Retorna (dados, origem)
    origem = 'cache' ou 'api'
    """
    cnpj_limpo = limpar_cnpj(cnpj)

    if len(cnpj_limpo) != 14:
        raise ValueError("CNPJ inválido: precisa ter 14 dígitos.")

    # 1) tenta cache
    registro = carregar_cache(cnpj_limpo)
    if registro and cache_esta_valido(registro):
        return registro, "cache"

    # 2) chama API
    dados_api = consultar_api_cnpjws(cnpj_limpo)
    salvar_cache(cnpj_limpo, dados_api)

    # recarrega pra ter o mesmo formato da função de cache
    registro_novo = carregar_cache(cnpj_limpo)
    return registro_novo, "api"


# ===========================
# STREAMLIT UI
# ===========================

def main():
    st.set_page_config(page_title="Consulta IE por CNPJ (CNPJ.ws)", page_icon="🧾")

    st.title("🧾 Consulta IE por CNPJ")
    st.write(
        "Aplicação utilizando a **API pública do CNPJ.ws** (limite de 3 consultas por minuto) "
        "com **cache local em SQLite** válido por ~2 meses."
    )

    cnpj_input = st.text_input("CNPJ", value="", placeholder="Digite o CNPJ do cliente")
    st.caption("Pode digitar com ou sem máscara. Ex: 12.345.678/0001-90 ou 12345678000190")

    if st.button("Consultar IE"):
        if not cnpj_input.strip():
            st.warning("Informe um CNPJ para consultar.")
            return

        try:
            with st.spinner("Consultando / lendo cache..."):
                dados, origem = obter_dados_cnpj(cnpj_input)

            cnpj_formatado = formatar_cnpj(dados["cnpj"])

            if origem == "cache":
                st.success("Resultado carregado do cache local ✅")
            else:
                st.success("Resultado obtido da API agora ✅ (pode levar alguns segundos)")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**CNPJ:** {cnpj_formatado}")
                st.write(f"**Razão Social:** {dados.get('razao_social') or '—'}")
            with col2:
                st.write(f"**IE principal:** `{dados.get('ie_principal') or '—'}`")
                st.write(f"**Atualizado na API em:** {dados.get('atualizado_em_api') or '—'}")

            if dados.get("atualizado_em_cache"):
                st.caption(
                    f"Salvo/atualizado no cache em: {dados['atualizado_em_cache'].strftime('%d/%m/%Y %H:%M:%S')}"
                )

            # Tabela com todas as IEs
            ies = dados.get("ufs_ies") or []
            if ies:
                st.subheader("Inscrições Estaduais encontradas")
                st.table(
                    [
                        {
                            "UF": item.get("uf"),
                            "IE": item.get("inscricao_estadual"),
                            "Ativo": "Sim" if item.get("ativo") else "Não",
                            "Atualizado em": item.get("atualizado_em"),
                        }
                        for item in ies
                    ]
                )
            else:
                st.info("Nenhuma inscrição estadual encontrada para este CNPJ na API.")

        except Exception as e:
            st.error(f"Erro: {e}")


if __name__ == "__main__":
    main()
