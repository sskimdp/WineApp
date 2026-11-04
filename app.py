import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

FEATURE_NAMES = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь, %",
    "is_red": "Красное вино",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def test_proba(_model, X):
    # подчёркивание: Streamlit не хеширует модель
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def load_threshold():
    return json.loads(CONFIG_PATH.read_text())["threshold"]


for path in [MODEL_PATH, CONFIG_PATH, TEST_PATH]:
    if not path.exists():
        st.error(f"Нет файла {path.name}. Сначала выполните ноутбук и сохраните модель.")
        st.stop()

model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test["target"]
proba_test = test_proba(model, X_test)

st.title("🍷 Подбор хороших вин")
st.caption("Логистическая регрессия по результатам лабораторного анализа вина.")

threshold = st.sidebar.slider(
    "Порог «хорошее вино»", 0.0, 1.0, float(load_threshold()), step=0.01, key="threshold"
)
st.sidebar.write(f"Вино попадает в подборку, если вероятность ≥ {threshold:.2f}.")

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

# 4.1
with tab_one:
    st.subheader("Оценка одного вина")
    values = {}
    left, right = st.columns(2)
    numeric = [f for f in features if f != "is_red"]
    for i, feature in enumerate(numeric):
        column = left if i % 2 == 0 else right
        values[feature] = column.number_input(
            FEATURE_NAMES.get(feature, feature),
            min_value=float(test[feature].min()),
            max_value=float(test[feature].max()),
            value=float(test[feature].median()),
            format="%.4f" if feature == "density" else "%.3f",
        )
    color = st.radio("Цвет вина", ["белое", "красное"], horizontal=True)
    values["is_red"] = 1 if color == "красное" else 0

    wine = pd.DataFrame([values])[features]
    proba = model.predict_proba(wine)[0, 1]
    st.metric("Вероятность, что вино хорошее", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= threshold:
        st.success("Берём в подборку: модель считает вино хорошим.")
    else:
        st.warning("Не берём: модель считает вино посредственным.")

# 4.2
with tab_metrics:
    st.subheader("Качество на тестовой выборке при текущем пороге")
    pred_test = (proba_test >= threshold).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics.precision_score(y_test, pred_test, zero_division=0):.3f}")
    c2.metric("Recall", f"{metrics.recall_score(y_test, pred_test):.3f}")
    c3.metric("F1", f"{metrics.f1_score(y_test, pred_test):.3f}")
    c4.metric("В подборке вин", int(pred_test.sum()))

    cm = metrics.confusion_matrix(y_test, pred_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["не хорошее", "хорошее"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Матрица ошибок")

    fpr, tpr, _ = metrics.roc_curve(y_test, proba_test)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y_test, proba_test):.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].scatter(fp / (fp + tn), tp / (tp + fn), color="red", s=80, zorder=3, label="текущий порог")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR (recall)")
    axes[1].set_title("ROC-кривая")
    axes[1].legend()

    prec, rec, _ = metrics.precision_recall_curve(y_test, proba_test)
    axes[2].plot(rec, prec, label=f"PR-AUC = {metrics.average_precision_score(y_test, proba_test):.3f}")
    axes[2].axhline(y_test.mean(), color="k", linestyle="--", alpha=0.5, label="случайная модель")
    if tp + fp > 0:
        axes[2].scatter(tp / (tp + fn), tp / (tp + fp), color="red", s=80, zorder=3, label="текущий порог")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("PR-кривая")
    axes[2].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Для сомелье. Если порог увеличить, модель станет строже: в подборку попадёт меньше вин, "
        "и среди них будет больше действительно хороших (растёт precision), но часть хороших вин "
        "она пропустит (падает recall). Если порог уменьшить, подборка станет шире и будет реже "
        "пропускать хорошие вина, но в неё чаще будут попадать посредственные."
    )

# 4.3
with tab_batch:
    st.subheader("Пакетное предсказание")
    st.write(f"Используется порог из боковой панели: {threshold:.2f}.")
    uploaded = st.file_uploader("CSV с признаками вин (без столбца target)", type="csv")
    if uploaded is None:
        st.info("Загрузите CSV-файл с колонками: " + ", ".join(features))
    else:
        try:
            batch = pd.read_csv(uploaded)
        except Exception:
            batch = None
            st.error("Не удалось прочитать файл. Проверьте, что это CSV с разделителем-запятой.")

        batch_proba = None
        if batch is not None:
            missing = [f for f in features if f not in batch.columns]
            if missing:
                st.error("В файле не хватает столбцов: " + ", ".join(missing))
            else:
                try:
                    batch_proba = model.predict_proba(batch[features])[:, 1]
                except Exception:
                    st.error("Не удалось сделать прогноз: проверьте, что все значения числовые и без пропусков.")

        if batch_proba is not None:
            result = batch.copy()
            result["proba_good"] = batch_proba.round(4)
            result["in_selection"] = (batch_proba >= threshold).astype(int)
            st.metric("Попадут в подборку", f"{int(result['in_selection'].sum())} из {len(result)}")
            st.dataframe(result.sort_values("proba_good", ascending=False), hide_index=True)
            st.download_button(
                "Скачать прогнозы",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="wine_predictions.csv",
                mime="text/csv",
                on_click="ignore",
            )

# 4.4
with tab_coef:
    st.subheader("Коэффициенты логистической регрессии")
    st.write(
        "Признаки стандартизированы, поэтому коэффициенты можно сравнивать между собой: "
        "чем больше коэффициент по модулю, тем сильнее признак влияет на решение."
    )
    coef = pd.Series(model[-1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([FEATURE_NAMES.get(f, f) for f in coef.index], coef.values,
            color=["tab:red" if v < 0 else "tab:green" for v in coef.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент (после стандартизации)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    top_for = [FEATURE_NAMES.get(f, f) for f in coef.sort_values(ascending=False).index[:3]]
    top_against = [FEATURE_NAMES.get(f, f) for f in coef.index[:3]]
    st.success("Сильнее всего за хорошее вино «голосуют»: " + ", ".join(top_for) + ".")
    st.error("Сильнее всего против «голосуют»: " + ", ".join(top_against) + ".")