import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import Tuple

st.set_page_config(page_title="Forecast Pallets 2026", layout="wide")

# ===================== Utils =====================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # normaliza nomes mantendo compatibilidade com os nomes pedidos
    new_cols = {}
    for c in df.columns:
        cl = c.strip().lower()
        # mapeia "toral por pallet" -> "total por pallet"
        if "toral" in cl and "pallet" in cl:
            new_cols[c] = "TOTAL POR PALLET"
        elif "pallet" in cl and "unico" in cl:
            new_cols[c] = "PALLET UNICO"
        elif "triagem" in cl and "unico" in cl:
            new_cols[c] = "DATA TRIAGEM UNICO"
        elif "receb" in cl and "unico" in cl:
            new_cols[c] = "DATA RECEBIMENTO UNICO"
        elif "total" in cl and "pallet" in cl:
            new_cols[c] = "TOTAL POR PALLET"
        else:
            new_cols[c] = c  # mantém original
    df = df.rename(columns=new_cols)
    return df

def _read_any(file) -> pd.DataFrame:
    fname = file.name if hasattr(file, "name") else str(file)
    if fname.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)

def _build_arrival_date(df: pd.DataFrame) -> pd.Series:
    col_rec = "DATA RECEBIMENTO UNICO"
    col_tri = "DATA TRIAGEM UNICO"
    # converte datas
    for c in [col_rec, col_tri]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        else:
            df[c] = pd.NaT

    mask_both = df[col_rec].notna() & df[col_tri].notna()
    if mask_both.any():
        atraso = (df.loc[mask_both, col_tri] - df.loc[mask_both, col_rec]).dt.days
        atraso = atraso.clip(lower=0)
        atraso_med = int(np.median(atraso))
    else:
        atraso_med = 0

    chegada = df[col_rec].copy()
    faltando = chegada.isna() & df[col_tri].notna()
    chegada.loc[faltando] = df.loc[faltando, col_tri] - pd.to_timedelta(atraso_med, unit="D")
    return pd.to_datetime(chegada)

def _weekly_agg(
    df: pd.DataFrame,
    target_mode: str,
    freq: str = "W-MON"
) -> Tuple[pd.DataFrame, str]:
    """target_mode: 'pallets' ou 'materiais' """
    df = df.copy()
    df["data_chegada"] = _build_arrival_date(df)
    df = df.loc[df["data_chegada"].notna()].copy()

    # linha do tempo contínua de 2023-12 até 2025-10 (ajusta automático pelos dados)
    start = (df["data_chegada"].min().normalize() if df["data_chegada"].notna().any()
             else pd.Timestamp("2024-01-01"))
    end = (df["data_chegada"].max().normalize()
           if df["data_chegada"].notna().any()
           else pd.Timestamp("2025-12-31"))

    # garante colunas esperadas
    if "PALLET UNICO" not in df.columns:
        raise ValueError("Coluna 'PALLET UNICO' não encontrada.")
    if target_mode == "materiais":
        if "TOTAL POR PALLET" not in df.columns:
            raise ValueError("Coluna 'TOTAL POR PALLET' não encontrada para soma de materiais.")
        df["TOTAL POR PALLET"] = pd.to_numeric(df["TOTAL POR PALLET"], errors="coerce").fillna(0)

    if target_mode == "pallets":
        agg = (df
               .groupby(pd.Grouper(key="data_chegada", freq=freq))
               .agg(y=("PALLET UNICO", "nunique"))
               .reset_index()
               .rename(columns={"data_chegada": "ds"}))
        label = "Pallets (nº únicos por semana)"
    else:
        agg = (df
               .groupby(pd.Grouper(key="data_chegada", freq=freq))
               .agg(y=("TOTAL POR PALLET", "sum"))
               .reset_index()
               .rename(columns={"data_chegada": "ds"}))
        label = "Materiais (soma total por semana)"

    # linha do tempo contínua
    timeline = pd.date_range(start, end, freq=freq)
    out = pd.DataFrame({"ds": timeline}).merge(agg, on="ds", how="left")
    out["y"] = out["y"].fillna(0.0)
    return out, label

def _compute_fh(last_ds: pd.Timestamp, end_year: int = 2026, freq: str = "W-MON") -> Tuple[int, pd.DatetimeIndex]:
    start = (last_ds + pd.offsets.Week(weekday=0))  # próxima segunda
    end = pd.Timestamp(f"{end_year}-12-31")
    future_idx = pd.date_range(start, end, freq=freq)
    return len(future_idx), future_idx

def _download_csv(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV", data=csv, file_name=filename, mime="text/csv")

# ============== Sidebar / Inputs ==============
st.sidebar.title("Configuração")
st.sidebar.markdown("Carregue **ANALISE_PSD** (CSV ou Excel) com as colunas:\n\n"
                    "- PALLET UNICO\n- DATA TRIAGEM UNICO\n- DATA RECEBIMENTO UNICO\n- TORAL POR PALLET (aceito)\n")
target_mode = st.sidebar.radio("Alvo da previsão", ["pallets", "materiais"], index=0,
                               help="pallets = contagem de pallets por semana; materiais = soma TOTAL_POR_PALLET por semana")
uploaded = st.sidebar.file_uploader("Enviar arquivo (ANALISE_PSD)", type=["csv", "xlsx", "xls"])
run_btn = st.sidebar.button("Treinar e Prever 2026")

st.title("🔮 Previsão 2026 • Pallets / Materiais (semanal)")
st.caption("Imputação de datas (recebimento prioritário; triagem − mediana do atraso como fallback) + AutoML (PyCaret).")

# ============== Main ==============
if uploaded is None:
    st.info("Envie o arquivo **ANALISE_PSD** na barra lateral para começar.")
    st.stop()

# Leitura e normalização
try:
    raw = _read_any(uploaded)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

raw = _normalize_cols(raw)

# Preview
with st.expander("🔎 Prévia dos dados (primeiras linhas)"):
    st.dataframe(raw.head(20), use_container_width=True)

# Agregação semanal
try:
    wk, y_label = _weekly_agg(raw, target_mode=target_mode, freq="W-MON")
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Início histórico", wk["ds"].min().date())
col2.metric("Fim histórico", wk["ds"].max().date())
col3.metric("Semanas no histórico", len(wk))

st.subheader("📈 Série semanal (histórico)")
st.line_chart(wk.set_index("ds")["y"])

if not run_btn:
    st.stop()

# ===================== Modelagem =====================
with st.spinner("Treinando modelos..."):
    best_name = None
    forecast_df = None
    metrics_note = ""
    fh, future_idx = _compute_fh(wk["ds"].max(), end_year=2026, freq="W-MON")

    # tenta PyCaret; se falhar, fallback
    try:
        from pycaret.time_series import TSForecastingExperiment

        exp = TSForecastingExperiment()
        # PyCaret espera DataFrame com coluna target ou Series
        data_ts = wk[["ds", "y"]].set_index("ds").asfreq("W-MON")["y"]

        exp.setup(
            data=data_ts,
            fh=fh,
            fold=3,
            session_id=42,
            verbose=False,
            seasonal_period="auto",
            imputation_method="ffill",
            transform_target=True,
        )
        best = exp.compare_models(sort="MASE", n_select=1, turbo=True)
        best_name = str(best)

        final_best = exp.finalize_model(best)
        preds = exp.predict_model(final_best, fh=fh)  # retorna série/df
        # Preds vira uma Series com index futuro; mapeia para DataFrame
        if isinstance(preds, pd.Series):
            f = preds.rename("p50").to_frame()
        else:
            # tenta colunas padrão
            if "y_pred" in preds.columns:
                f = preds[["y_pred"]].rename(columns={"y_pred": "p50"})
            elif "Label" in preds.columns:
                f = preds[["Label"]].rename(columns={"Label": "p50"})
            else:
                # última coluna como previsão
                f = preds.iloc[:, [-1]].rename(columns={preds.columns[-1]: "p50"})

        f.index = pd.to_datetime(f.index)
        f = f.reset_index().rename(columns={"index": "ds"})
        # Garante apenas 2026
        f = f[(f["ds"] >= pd.Timestamp("2026-01-01")) & (f["ds"] <= pd.Timestamp("2026-12-31"))]
        forecast_df = f.copy()
        metrics_note = "Modelo selecionado via PyCaret (melhor por MASE)."

    except Exception as e:
        # Fallback simples: sazonal por semana do ano (média histórica da mesma semana)
        metrics_note = f"PyCaret indisponível ({e}). Usando fallback sazonal por semana do ano."
        hist = wk.copy()
        hist["week"] = hist["ds"].dt.isocalendar().week.astype(int)
        seasonal = hist.groupby("week")["y"].mean()

        fut = pd.DataFrame({"ds": future_idx})
        fut["week"] = fut["ds"].dt.isocalendar().week.astype(int)
        fut["p50"] = fut["week"].map(seasonal).fillna(hist["y"].mean())
        forecast_df = fut[["ds", "p50"]].copy()
        best_name = "Sazonal (média por semana do ano)"

# ===================== Resultados =====================
st.subheader("Modelo escolhido")
st.write(best_name)
st.caption(metrics_note)

hist_2024_2025 = wk.copy()
hist_2024_2025["tipo"] = "Histórico"
fc_plot = forecast_df.copy()
fc_plot["tipo"] = "Previsão 2026 (p50)"

plot_df = pd.concat([
    hist_2024_2025.rename(columns={"y": "valor"})[["ds", "valor", "tipo"]],
    fc_plot.rename(columns={"p50": "valor"})[["ds", "valor", "tipo"]]
], ignore_index=True)

st.subheader(f"📊 {y_label}")
st.line_chart(plot_df.pivot(index="ds", columns="tipo", values="valor"))

# Agregado mensal
forecast_df["ano"] = forecast_df["ds"].dt.year
forecast_df["mes"] = forecast_df["ds"].dt.month
mensal = (forecast_df.groupby(["ano", "mes"], as_index=False)
          .agg(p50=("p50", "sum")))
total_2026 = float(forecast_df["p50"].sum())

c1, c2 = st.columns(2)
with c1:
    st.markdown("### 📦 Previsão semanal 2026 (p50)")
    st.dataframe(forecast_df, use_container_width=True)
    _download_csv(forecast_df, f"forecast_2026_{target_mode}_semanal.csv")
with c2:
    st.markdown("### 🗓️ Previsão mensal 2026 (p50)")
    st.dataframe(mensal, use_container_width=True)
    _download_csv(mensal, f"forecast_2026_{target_mode}_mensal.csv")

st.success(f"TOTAL 2026 (p50): {total_2026:,.0f}")

st.markdown("---")
st.caption("Dica: se quiser rodar por **segmento/UF** no futuro, basta filtrar e repetir o pipeline para cada grupo, depois somar.")

