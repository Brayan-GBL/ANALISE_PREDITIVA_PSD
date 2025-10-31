import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple, List, Optional
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

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
    data_chegada: prioriza RECEBIMENTO; se faltar e existir TRIAGEM,
    usa TRIAGEM - mediana(max(TRIAGEM-RECEBIMENTO,0)).
    """
    col_rec = "DATA RECEBIMENTO UNICO"
    col_tri = "DATA TRIAGEM UNICO"
    for c in [col_rec, col_tri]:
        if c not in df.columns:
            df[c] = pd.NaT
        df[c] = pd.to_datetime(df[c], errors="coerce")

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

def _weekly_agg(df: pd.DataFrame, target_mode: str, freq: str = "W-MON") -> Tuple[pd.DataFrame, str]:
    """
    Retorna série semanal: [ds, y]
      - pallets: nº de PALLET UNICO distintos
      - materiais: soma de TOTAL POR PALLET
    """
    df = df.copy()
    df["data_chegada"] = _build_arrival_date(df)
    df = df.loc[df["data_chegada"].notna()].copy()

    if "PALLET UNICO" not in df.columns:
        raise ValueError("Coluna 'PALLET UNICO' não encontrada no arquivo.")
    if target_mode == "materiais":
        if "TOTAL POR PALLET" not in df.columns:
            raise ValueError("Coluna 'TOTAL POR PALLET' não encontrada.")
        df["TOTAL POR PALLET"] = pd.to_numeric(df["TOTAL POR PALLET"], errors="coerce").fillna(0)

    if target_mode == "pallets":
        agg = (df.groupby(pd.Grouper(key="data_chegada", freq=freq))
                 .agg(y=("PALLET UNICO", "nunique"))
                 .reset_index()
                 .rename(columns={"data_chegada": "ds"}))
        label = "Pallets (nº únicos por semana)"
    else:
        agg = (df.groupby(pd.Grouper(key="data_chegada", freq=freq))
                 .agg(y=("TOTAL POR PALLET", "sum"))
                 .reset_index()
                 .rename(columns={"data_chegada": "ds"}))
        label = "Materiais (soma por semana)"

    if len(agg) == 0:
        return pd.DataFrame(columns=["ds", "y"]), label

    start = agg["ds"].min().normalize()
    end = agg["ds"].max().normalize()
    timeline = pd.date_range(start, end, freq=freq)
    out = pd.DataFrame({"ds": timeline}).merge(agg, on="ds", how="left")
    out["y"] = out["y"].fillna(0.0)
    return out, label

def _compute_future_index(last_ds: pd.Timestamp, end_year: int = 2026, freq: str = "W-MON") -> pd.DatetimeIndex:
    """Índice semanal de (próxima segunda) até 31/12/end_year; filtra só 2026."""
    if pd.isna(last_ds):
        start = pd.Timestamp(f"{end_year}-01-01")
    else:
        start = (last_ds + pd.offsets.Week(weekday=0))
    end = pd.Timestamp(f"{end_year}-12-31")
    future_idx = pd.date_range(start, end, freq=freq)
    future_idx = future_idx[(future_idx >= pd.Timestamp("2026-01-01")) & (future_idx <= pd.Timestamp("2026-12-31"))]
    return future_idx

def _download_csv(df: pd.DataFrame, filename: str):
    st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")

# ===================== Baseline (média por semana do ano) =====================
def _seasonal_fallback(wk_df: pd.DataFrame, future_idx: pd.DatetimeIndex):
    hist = wk_df.copy()
    hist["week"] = hist["ds"].dt.isocalendar().week.astype(int)
    seasonal = hist.groupby("week")["y"].mean()
    fut = pd.DataFrame({"ds": future_idx})
    fut["week"] = fut["ds"].dt.isocalendar().week.astype(int)
    fut["p50"] = fut["week"].map(seasonal).fillna(hist["y"].mean())
    return fut[["ds", "p50"]]

# ===================== Features p/ modelo =====================
def _select_lags(n_points: int) -> List[int]:
    base = [1, 2, 4, 8, 12, 26, 52]
    return [L for L in base if L < n_points]

def _make_features(wk_df: pd.DataFrame, lags: List[int], rolls: Optional[List[int]] = None) -> pd.DataFrame:
    df = wk_df.copy().sort_values("ds").reset_index(drop=True)
    df["week"] = df["ds"].dt.isocalendar().week.astype(int)
    df["month"] = df["ds"].dt.month.astype(int)
    for L in lags:
        df[f"lag_{L}"] = df["y"].shift(L)
    if rolls:
        for W in rolls:
            df[f"rollmean_{W}"] = df["y"].shift(1).rolling(W).mean()
    return df.dropna().reset_index(drop=True)

def _add_fourier(df, date_col="ds", period=52, K=2):
    df = df.copy()
    w = df[date_col].dt.isocalendar().week.astype(int)
    for k in range(1, K+1):
        df[f"fourier_sin_{k}"] = np.sin(2*np.pi*k*w/period)
        df[f"fourier_cos_{k}"] = np.cos(2*np.pi*k*w/period)
    return df

# ===================== Modelo (sklearn) =====================
def _forecast_sklearn(wk_df: pd.DataFrame, future_idx: pd.DatetimeIndex, target_mode: str):
    """
    HistGradientBoosting com:
      - loss='poisson' para pallets, 'squared_error' para materiais
      - cap de outliers (p99)
      - Fourier (sazonalidade semanal)
      - previsão recursiva estável mantendo MESMO espaço de features
    """
    n = len(wk_df)
    lags_all = _select_lags(n) or [1]
    rolls_all = [x for x in [4, 8, 12, 26] if x < n]

    feat = _make_features(wk_df, lags_all, rolls_all)
    if feat.empty:
        return _seasonal_fallback(wk_df, future_idx), None

    feat = _add_fourier(feat, date_col="ds", period=52, K=2)

    y = feat["y"].astype(float).values
    p_hi = np.percentile(y, 99) if len(y) >= 20 else y.max()
    y_cap = np.clip(y, 0, p_hi)

    feature_cols = [c for c in feat.columns if c not in ["ds", "y"]]
    X = feat[feature_cols].astype(float).values

    # CV temporal (com redução automática de splits)
    maes = []
    splits = 4
    while splits > 1:
        try:
            tscv = TimeSeriesSplit(n_splits=splits)
            for tr, te in tscv.split(X):
                loss = "poisson" if target_mode == "pallets" else "squared_error"
                m = HistGradientBoostingRegressor(random_state=42, loss=loss)
                m.fit(X[tr], y_cap[tr])
                p = m.predict(X[te])
                maes.append(mean_absolute_error(y_cap[te], p))
            break
        except Exception:
            splits -= 1
    mae_cv = float(np.median(maes)) if maes else None

    # Treino final
    loss = "poisson" if target_mode == "pallets" else "squared_error"
    final = HistGradientBoostingRegressor(random_state=42, loss=loss)
    final.fit(X, y_cap)

    # Previsão recursiva respeitando feature_cols
    hist_for_fc = feat[["ds", "y"]].copy()
    rows = []
    for d in future_idx:
        s = hist_for_fc["y"].astype(float)
        curr_len = len(s)
        last_val = float(s.iloc[-1]) if curr_len else 0.0

        row = {"week": int(d.isocalendar().week), "month": int(d.month)}
        # preencher TODAS as features na MESMA ORDEM
        row_feat = {}
        for col in feature_cols:
            if col == "week":
                row_feat[col] = row["week"]
            elif col == "month":
                row_feat[col] = row["month"]
            elif col.startswith("lag_"):
                try:
                    L = int(col.split("_", 1)[1])
                except Exception:
                    L = 1
                row_feat[col] = float(s.iloc[-L]) if curr_len >= L else last_val
            elif col.startswith("rollmean_"):
                try:
                    W = int(col.split("_", 1)[1])
                except Exception:
                    W = min(4, curr_len or 1)
                w = min(W, curr_len)
                row_feat[col] = float(s.iloc[-w:].mean()) if w > 0 else last_val
            elif col.startswith("fourier_sin_"):
                k = int(col.split("_")[-1])
                row_feat[col] = np.sin(2*np.pi*k*row["week"]/52)
            elif col.startswith("fourier_cos_"):
                k = int(col.split("_")[-1])
                row_feat[col] = np.cos(2*np.pi*k*row["week"]/52)
            else:
                row_feat[col] = last_val

        Xn = np.array([[row_feat[c] for c in feature_cols]], dtype=float)
        yhat = float(final.predict(Xn))
        yhat = max(0.0, yhat)
        rows.append({"ds": d, "p50": yhat})
        hist_for_fc = pd.concat([hist_for_fc, pd.DataFrame([{"ds": d, "y": yhat}])], ignore_index=True)

    fc = pd.DataFrame(rows)
    return fc, mae_cv

# ===================== Sidebar / Inputs =====================
st.sidebar.title("Configuração")
st.sidebar.markdown(
    "Envie **ANALISE_PSD** (CSV/Excel) com as colunas:\n"
    "- **PALLET UNICO**\n- **DATA TRIAGEM UNICO**\n- **DATA RECEBIMENTO UNICO**\n- **TOTAL POR PALLET**"
)
target_mode = st.sidebar.radio("Alvo da previsão", ["pallets", "materiais"], index=0)
uploaded = st.sidebar.file_uploader("Enviar arquivo (ANALISE_PSD)", type=["csv", "xlsx", "xls"])

# Botões
train_btn = st.sidebar.button("🚀 Treinar / Atualizar previsão 2026", type="primary")
show_baseline = st.sidebar.checkbox("➕ Mostrar baseline sazonal (média por semana do ano)", value=False)

st.title("Previsão 2026 • Pallets / Materiais (semanal)")
st.caption("Imputação de datas: Recebimento prioritário; se ausente, usa Triagem − mediana do atraso (triagem−recebimento).")

# ===================== Main =====================
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

# Métricas
start_ds = wk["ds"].min() if len(wk) else pd.NaT
end_ds = wk["ds"].max() if len(wk) else pd.NaT
start_str = start_ds.strftime("%Y-%m-%d") if pd.notna(start_ds) else "—"
end_str = end_ds.strftime("%Y-%m-%d") if pd.notna(end_ds) else "—"
n_semanas = int(len(wk)) if len(wk) else 0

c1, c2, c3 = st.columns(3)
c1.metric("Início histórico", start_str)
c2.metric("Fim histórico", end_str)
c3.metric("Semanas no histórico", f"{n_semanas:,}".replace(",", "."))

st.subheader("📈 Série semanal (histórico)")
if not len(wk):
    st.info("Sem dados no histórico para plotar.")
    st.stop()

# ========== Treinamento (persistente) ==========
future_idx = _compute_future_index(wk["ds"].max(), end_year=2026, freq="W-MON")

if train_btn or "model_result" not in st.session_state or st.session_state.get("last_target_mode") != target_mode:
    with st.spinner("Treinando e gerando previsão..."):
        fc_model, mae_cv = _forecast_sklearn(wk, future_idx, target_mode)
        st.session_state.model_result = {
            "forecast_df": fc_model,
            "mae_cv": mae_cv,
            "best_name": "HistGradientBoosting (scikit-learn) com Fourier",
            "metrics_note": f"Backtest MAE (mediano): {mae_cv:.2f}" if mae_cv is not None else "Backtest indisponível (histórico curto).",
        }
        st.session_state.last_target_mode = target_mode

# Sempre disponível após treinar pelo menos 1x
fc_model = st.session_state.get("model_result", {}).get("forecast_df")
mae_cv = st.session_state.get("model_result", {}).get("mae_cv")
best_name = st.session_state.get("model_result", {}).get("best_name", "—")
metrics_note = st.session_state.get("model_result", {}).get("metrics_note", "—")

# Baseline opcional (não substitui o modelo; é só comparação)
fc_base = _seasonal_fallback(wk, future_idx) if show_baseline else None

# ========== Gráficos (com rótulos para todas as séries) ==========
hist_df = wk.copy()
hist_df["tipo"] = "Histórico"
hist_df = hist_df.rename(columns={"y": "valor"})[["ds", "valor", "tipo"]]

series_to_concat = [hist_df]

if isinstance(fc_model, pd.DataFrame) and len(fc_model):
    tmp = fc_model.copy()
    tmp["tipo"] = "Previsão 2026 (p50)"
    tmp = tmp.rename(columns={"p50": "valor"})[["ds", "valor", "tipo"]]
    series_to_concat.append(tmp)

if isinstance(fc_base, pd.DataFrame) and len(fc_base):
    tmpb = fc_base.copy()
    tmpb["tipo"] = "Baseline sazonal (p50)"
    tmpb = tmpb.rename(columns={"p50": "valor"})[["ds", "valor", "tipo"]]
    series_to_concat.append(tmpb)

plot_df = pd.concat(series_to_concat, ignore_index=True)
plot_df["valor_fmt"] = plot_df["valor"].round(0)
plot_df = plot_df.sort_values("ds")
plot_df["idx"] = plot_df.groupby("tipo").cumcount()

st.subheader(f"📊 {y_label}")

with st.expander("⚙️ Opções do gráfico"):
    # Guardar opções no estado para não resetar após interação
    if "opt_points" not in st.session_state: st.session_state.opt_points = True
    if "opt_labels" not in st.session_state: st.session_state.opt_labels = True
    if "opt_label_every" not in st.session_state: st.session_state.opt_label_every = 4
    if "opt_label_scope" not in st.session_state: st.session_state.opt_label_scope = "Todos"

    st.session_state.opt_points = st.checkbox("Mostrar marcadores nos pontos", value=st.session_state.opt_points)
    st.session_state.opt_labels = st.checkbox("Mostrar valores como rótulos", value=st.session_state.opt_labels)
    st.session_state.opt_label_every = st.number_input("Exibir rótulo a cada N pontos", min_value=1, max_value=52,
                                                       value=st.session_state.opt_label_every, step=1)
    st.session_state.opt_label_scope = st.selectbox(
        "Séries com rótulo",
        ["Todos", "Histórico", "Previsão 2026 (p50)", "Baseline sazonal (p50)"],
        index=["Todos","Histórico","Previsão 2026 (p50)","Baseline sazonal (p50)"].index(st.session_state.opt_label_scope)
    )

# Filtragem para rótulos
every = int(st.session_state.opt_label_every)
if st.session_state.opt_label_scope == "Todos":
    mask_lbl = (plot_df["idx"] % every == 0)
else:
    mask_lbl = (plot_df["tipo"] == st.session_state.opt_label_scope) & (plot_df["idx"] % every == 0)
plot_labels = plot_df.loc[mask_lbl].copy()

base = alt.Chart(plot_df).encode(
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
points = base.mark_point() if st.session_state.opt_points else alt.Chart()

if st.session_state.opt_labels:
    text = alt.Chart(plot_labels).mark_text(align='left', dx=5, dy=-5).encode(
        x='ds:T', y='valor:Q', text=alt.Text('valor_fmt:Q'), color='tipo:N'
    )
    chart = (line + points + text).interactive()
else:
    chart = (line + points).interactive()

st.altair_chart(chart, use_container_width=True)

# ========== Bloco de resultados ==========
if isinstance(fc_model, pd.DataFrame) and len(fc_model):
    st.subheader("Modelo escolhido")
    st.write(best_name)
    st.caption(metrics_note)

    # Tabelas e downloads
    fc = fc_model.copy()
    fc["ano"] = fc["ds"].dt.year
    fc["mes"] = fc["ds"].dt.month
    mensal = fc.groupby(["ano", "mes"], as_index=False).agg(p50=("p50", "sum"))
    total_2026 = float(fc["p50"].sum())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📦 Previsão semanal 2026 (p50)")
        st.dataframe(fc, use_container_width=True)
        _download_csv(fc, f"forecast_2026_{target_mode}_semanal.csv")
    with c2:
        st.markdown("### 🗓️ Previsão mensal 2026 (p50)")
        st.dataframe(mensal, use_container_width=True)
        _download_csv(mensal, f"forecast_2026_{target_mode}_mensal.csv")

    st.success(f"TOTAL 2026 (p50): {total_2026:,.0f}")

# Baseline resumo (se exibida)
if isinstance(fc_base, pd.DataFrame) and len(fc_base):
    st.info("Baseline exibida: Média por semana do ano calculada a partir do histórico atual.")
