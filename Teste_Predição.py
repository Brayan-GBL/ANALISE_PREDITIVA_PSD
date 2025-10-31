import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from typing import Tuple

st.set_page_config(page_title="Forecast Pallets 2026", layout="wide")

# ===================== Utils =====================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes das colunas mantendo os nomes esperados."""
    new_cols = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "total" in cl and "pallet" in cl:
            new_cols[c] = "TOTAL POR PALLET"
        elif "pallet" in cl and "unico" in cl:
            new_cols[c] = "PALLET UNICO"
        elif "triagem" in cl and "unico" in cl:
            new_cols[c] = "DATA TRIAGEM UNICO"
        elif ("receb" in cl or "recebi" in cl) and "unico" in cl:
            new_cols[c] = "DATA RECEBIMENTO UNICO"
        else:
            new_cols[c] = c
    return df.rename(columns=new_cols)

def _read_any(file) -> pd.DataFrame:
    fname = file.name if hasattr(file, "name") else str(file)
    if fname.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)

def _build_arrival_date(df: pd.DataFrame) -> pd.Series:
    """
    Cria 'data_chegada' priorizando DATA RECEBIMENTO UNICO.
    Se faltar e houver DATA TRIAGEM UNICO, usa triagem - mediana_do_atraso(triagem - recebimento, truncado em 0).
    Se ambas faltarem, fica NaT (linha será descartada depois).
    """
    col_rec = "DATA RECEBIMENTO UNICO"
    col_tri = "DATA TRIAGEM UNICO"

    # garante colunas e converte pra datetime
    for c in [col_rec, col_tri]:
        if c not in df.columns:
            df[c] = pd.NaT
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # atraso mediano onde existem as duas datas (triagem - recebimento)
    mask_both = df[col_rec].notna() & df[col_tri].notna()
    if mask_both.any():
        atraso = (df.loc[mask_both, col_tri] - df.loc[mask_both, col_rec]).dt.days
        atraso = atraso.clip(lower=0)
        atraso_med = int(np.median(atraso))
    else:
        atraso_med = 0

    # imputação
    chegada = df[col_rec].copy()
    faltando = chegada.isna() & df[col_tri].notna()
    chegada.loc[faltando] = df.loc[faltando, col_tri] - pd.to_timedelta(atraso_med, unit="D")

    return pd.to_datetime(chegada)

def _weekly_agg(
    df: pd.DataFrame,
    target_mode: str,
    freq: str = "W-MON"
) -> Tuple[pd.DataFrame, str]:
    """
    Aggrega semanalmente:
      - pallets: nº de 'PALLET UNICO' distintos por semana
      - materiais: soma de 'TOTAL POR PALLET' por semana
    Retorna DataFrame com colunas [ds, y] e o label do alvo.
    """
    df = df.copy()
    df["data_chegada"] = _build_arrival_date(df)
    df = df.loc[df["data_chegada"].notna()].copy()

    if "PALLET UNICO" not in df.columns:
        raise ValueError("Coluna 'PALLET UNICO' não encontrada no arquivo.")
    if target_mode == "materiais":
        if "TOTAL POR PALLET" not in df.columns:
            raise ValueError("Coluna 'TOTAL POR PALLET' não encontrada para soma de materiais.")
        df["TOTAL POR PALLET"] = pd.to_numeric(df["TOTAL POR PALLET"], errors="coerce").fillna(0)

    # agregações
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
        label = "Materiais (soma por semana)"

    # linha do tempo contínua entre min e max
    if len(agg) == 0:
        return pd.DataFrame(columns=["ds", "y"]), label

    start = agg["ds"].min().normalize()
    end = agg["ds"].max().normalize()
    timeline = pd.date_range(start, end, freq=freq)
    out = pd.DataFrame({"ds": timeline}).merge(agg, on="ds", how="left")
    out["y"] = out["y"].fillna(0.0)
    return out, label

def _compute_future_index(last_ds: pd.Timestamp, end_year: int = 2026, freq: str = "W-MON") -> pd.DatetimeIndex:
    """Cria índice semanal de (próxima semana) até 31/12/end_year."""
    # próxima semana alinhada a segunda-feira
    if pd.isna(last_ds):
        start = pd.Timestamp(f"{end_year}-01-01")
    else:
        start = (last_ds + pd.offsets.Week(weekday=0))
    end = pd.Timestamp(f"{end_year}-12-31")
    future_idx = pd.date_range(start, end, freq=freq)
    # mantém apenas 2026
    future_idx = future_idx[(future_idx >= pd.Timestamp("2026-01-01")) & (future_idx <= pd.Timestamp("2026-12-31"))]
    return future_idx

def _download_csv(df: pd.DataFrame, filename: str):
    st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

# ============== Sidebar / Inputs ==============
st.sidebar.title("Configuração")
st.sidebar.markdown(
    "Envie **ANALISE_PSD** (CSV/Excel) com as colunas:\n"
    "- **PALLET UNICO**\n- **DATA TRIAGEM UNICO**\n- **DATA RECEBIMENTO UNICO**\n- **TOTAL POR PALLET**"
)
target_mode = st.sidebar.radio("Alvo da previsão", ["pallets", "materiais"], index=0)
uploaded = st.sidebar.file_uploader("Enviar arquivo (ANALISE_PSD)", type=["csv", "xlsx", "xls"])
run_btn = st.sidebar.button("Treinar e Prever 2026")

st.title("Previsão 2026 • Pallets / Materiais (semanal)")
st.caption("Imputação de datas: Recebimento prioritário; se ausente, usa Triagem − mediana do atraso (triagem−recebimento).")

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

with st.expander("🔎 Prévia dos dados"):
    st.dataframe(raw.head(30), use_container_width=True)

# Agregação semanal
try:
    wk, y_label = _weekly_agg(raw, target_mode=target_mode, freq="W-MON")
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")
    st.stop()

# --- Métricas (em STRING) ---
start_ds = wk["ds"].min() if len(wk) else pd.NaT
end_ds = wk["ds"].max() if len(wk) else pd.NaT
start_str = start_ds.strftime("%Y-%m-%d") if pd.notna(start_ds) else "—"
end_str = end_ds.strftime("%Y-%m-%d") if pd.notna(end_ds) else "—"
n_semanas = int(len(wk)) if len(wk) else 0

col1, col2, col3 = st.columns(3)
col1.metric("Início histórico", start_str)
col2.metric("Fim histórico", end_str)
col3.metric("Semanas no histórico", f"{n_semanas:,}".replace(",", "."))

st.subheader("📈 Série semanal (histórico)")
if not len(wk):
    st.info("Sem dados no histórico para plotar.")
    st.stop()

# ===================== Modelagem (opcional via botão) =====================
if run_btn:
    with st.spinner("Treinando modelos..."):
        best_name = None
        metrics_note = ""
        forecast_df = None

        future_idx = _compute_future_index(wk["ds"].max(), end_year=2026, freq="W-MON")
        fh = len(future_idx)

        try:
            # ---------- PyCaret ----------
            from pycaret.time_series import TSForecastingExperiment

            exp = TSForecastingExperiment()

            # Série semanal + imputação ANTES do setup
            series = wk.set_index("ds")["y"].asfreq("W-MON")
            series = series.ffill().fillna(0)

            exp.setup(
                data=series,
                fh=fh,
                fold=3,
                session_id=42,
                verbose=False,
                seasonal_period="auto",
                transform_target=True,
            )
            best = exp.compare_models(sort="MASE", n_select=1, turbo=True)
            best_name = str(best)
            final_best = exp.finalize_model(best)
            preds = exp.predict_model(final_best, fh=fh)

            # Normaliza saída
            if isinstance(preds, pd.Series):
                f = preds.rename("p50").to_frame()
            else:
                if "y_pred" in preds.columns:
                    f = preds[["y_pred"]].rename(columns={"y_pred": "p50"})
                elif "Label" in preds.columns:
                    f = preds[["Label"]].rename(columns={"Label": "p50"})
                else:
                    f = preds.iloc[:, [-1]].rename(columns={preds.columns[-1]: "p50"})
            f.index = pd.to_datetime(f.index)
            f = f.reset_index().rename(columns={"index": "ds"})
            f = f[(f["ds"] >= pd.Timestamp("2026-01-01")) & (f["ds"] <= pd.Timestamp("2026-12-31"))]
            forecast_df = f.copy()
            metrics_note = "Modelo selecionado via PyCaret (melhor por MASE)."

        except Exception as e:
            # ---------- Fallback: sazonalidade por semana do ano ----------
            metrics_note = f"PyCaret indisponível ({e}). Usando fallback sazonal (média por semana do ano)."
            hist = wk.copy()
            hist["week"] = hist["ds"].dt.isocalendar().week.astype(int)
            seasonal = hist.groupby("week")["y"].mean()
            fut = pd.DataFrame({"ds": future_idx})
            fut["week"] = fut["ds"].dt.isocalendar().week.astype(int)
            fut["p50"] = fut["week"].map(seasonal).fillna(hist["y"].mean())
            forecast_df = fut[["ds", "p50"]].copy()
            best_name = "Sazonal (média por semana do ano)"

    st.subheader("Modelo escolhido")
    st.write(best_name)
    st.caption(metrics_note)

    # Plot histórico + previsão (com Altair e rótulos)
    hist_df = wk.copy()
    hist_df["tipo"] = "Histórico"
    fc_plot = forecast_df.copy()
    fc_plot["tipo"] = "Previsão 2026 (p50)"
    plot_df = pd.concat([
        hist_df.rename(columns={"y": "valor"})[["ds", "valor", "tipo"]],
        fc_plot.rename(columns={"p50": "valor"})[["ds", "valor", "tipo"]],
    ], ignore_index=True)

    st.subheader(f"📊 {y_label}")

    with st.expander("⚙️ Opções do gráfico"):
        show_points = st.checkbox("Mostrar marcadores nos pontos", value=True)
        show_labels = st.checkbox("Mostrar valores como rótulos", value=True)
        label_every = st.number_input("Exibir rótulo a cada N pontos", min_value=1, max_value=52, value=4, step=1)
        label_only_2026 = st.checkbox("Rótulos apenas na previsão 2026", value=True)

    plot_df2 = plot_df.copy()
    plot_df2["valor_fmt"] = plot_df2["valor"].round(0)
    plot_df2 = plot_df2.sort_values("ds")
    plot_df2["idx"] = plot_df2.groupby("tipo").cumcount()

    if label_only_2026:
        mask_rotulo = (plot_df2["tipo"].str.contains("Previsão", na=False)) & (plot_df2["idx"] % label_every == 0)
    else:
        mask_rotulo = (plot_df2["idx"] % label_every == 0)

    plot_df_labels = plot_df2.loc[mask_rotulo].copy()

    base = alt.Chart(plot_df2).encode(
        x=alt.X('ds:T', title='Data'),
        y=alt.Y('valor:Q', title=y_label),
        color=alt.Color('tipo:N', title='Série'),
        tooltip=[
            alt.Tooltip('ds:T', title='Data'),
            alt.Tooltip('tipo:N', title='Série'),
            alt.Tooltip('valor_fmt:Q', title='Valor')
        ]
    )

    line = base.mark_line()
    points = base.mark_point() if show_points else alt.Chart()

    if show_labels:
        text = alt.Chart(plot_df_labels).mark_text(align='left', dx=5, dy=-5).encode(
            x='ds:T',
            y='valor:Q',
            text=alt.Text('valor_fmt:Q'),
            color='tipo:N'
        )
        chart = (line + points + text).interactive()
    else:
        chart = (line + points).interactive()

    st.altair_chart(chart, use_container_width=True)

    # Agregado mensal + total
    forecast_df["ano"] = forecast_df["ds"].dt.year
    forecast_df["mes"] = forecast_df["ds"].dt.month
    mensal = forecast_df.groupby(["ano", "mes"], as_index=False).agg(p50=("p50", "sum"))
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

else:
    # Se ainda não clicou em treinar, pelo menos mostre o histórico com Altair já com labels
    st.subheader(f"📊 {y_label} (histórico)")
    with st.expander("⚙️ Opções do gráfico"):
        show_points = st.checkbox("Mostrar marcadores nos pontos", value=True, key="hist_points")
        show_labels = st.checkbox("Mostrar valores como rótulos", value=False, key="hist_labels")
        label_every = st.number_input("Exibir rótulo a cada N pontos", min_value=1, max_value=52, value=4, step=1, key="hist_every")

    hist_plot = wk.copy().rename(columns={"y": "valor"})
    hist_plot["tipo"] = "Histórico"
    hist_plot["valor_fmt"] = hist_plot["valor"].round(0)
    hist_plot = hist_plot.sort_values("ds")
    hist_plot["idx"] = hist_plot.groupby("tipo").cumcount()
    mask_rotulo = (hist_plot["idx"] % label_every == 0)
    hist_labels = hist_plot.loc[mask_rotulo].copy()

    base_h = alt.Chart(hist_plot).encode(
        x=alt.X('ds:T', title='Data'),
        y=alt.Y('valor:Q', title=y_label),
        color=alt.Color('tipo:N', title='Série'),
        tooltip=[
            alt.Tooltip('ds:T', title='Data'),
            alt.Tooltip('valor_fmt:Q', title='Valor')
        ]
    )
    line_h = base_h.mark_line()
    points_h = base_h.mark_point() if show_points else alt.Chart()
    if show_labels:
        text_h = alt.Chart(hist_labels).mark_text(align='left', dx=5, dy=-5).encode(
            x='ds:T',
            y='valor:Q',
            text=alt.Text('valor_fmt:Q'),
            color='tipo:N'
        )
        chart_h = (line_h + points_h + text_h).interactive()
    else:
        chart_h = (line_h + points_h).interactive()

    st.altair_chart(chart_h, use_container_width=True)
