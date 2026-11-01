# ============================================================
# 🚀 DATA EXPLORER для /content/sample_data  (Google Colab)
# Вставь весь код в одну ячейку и запусти (Shift+Enter)
# ============================================================
import os, json, glob, random, datetime, warnings, html as html_lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML, Markdown

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")
pd.set_option("display.max_columns", 30)

DATA_DIR = os.environ.get("DATA_DIR", "/content/sample_data")

try:
    import ipywidgets as widgets
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ---------- 🛠 Утилиты ----------
def human_size(n):
    for unit in ["Б", "КБ", "МБ", "ГБ"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


def banner(text, emoji="✨", c1="#667eea", c2="#764ba2"):
    display(HTML(f"""
    <div style="background:linear-gradient(90deg,{c1},{c2});color:white;
                padding:12px 20px;border-radius:12px;margin:18px 0 8px 0;
                font-size:20px;font-weight:bold;font-family:sans-serif;
                box-shadow:0 4px 12px rgba(0,0,0,.2)">{emoji} {text}</div>"""))


def is_mnist(path, df=None):
    name = os.path.basename(path).lower()
    return "mnist" in name or (df is not None and df.shape[1] == 785)


def load_file(path):
    """Умная загрузка: CSV / JSON / текст."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            if is_mnist(path):
                return pd.read_csv(path, header=None)
            return pd.read_csv(path)
        if ext == ".json":
            try:
                return pd.read_json(path)
            except ValueError:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"⚠️ Не удалось прочитать: {e}"


# ---------- 📂 Загрузка всех файлов ----------
paths = sorted(p for p in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True)
               if os.path.isfile(p))
DATA = {os.path.basename(p): load_file(p) for p in paths}
TABLES = {k: v for k, v in DATA.items() if isinstance(v, pd.DataFrame)}

display(HTML(f"""
<div style="text-align:center;padding:25px;border-radius:16px;
            background:linear-gradient(135deg,#ff6a88,#ff99ac,#fcb69f);
            color:white;font-family:sans-serif">
  <div style="font-size:34px;font-weight:bold">🔍 Data Explorer</div>
  <div style="font-size:16px;margin-top:6px">Папка <code>{DATA_DIR}</code> ·
  найдено файлов: <b>{len(paths)}</b> · таблиц: <b>{len(TABLES)}</b> ·
  {datetime.datetime.now():%d.%m.%Y %H:%M}</div>
</div>"""))

if not paths:
    print("Папка пуста или не существует 😢")


# ---------- 1. 📋 Инвентаризация ----------
banner("Инвентаризация файлов", "📋")
ICONS = {".csv": "📊", ".json": "🧩", ".md": "📝", ".txt": "📄"}
rows = []
for p in paths:
    name = os.path.basename(p)
    obj = DATA[name]
    ext = os.path.splitext(p)[1].lower()
    shape = obj.shape if isinstance(obj, pd.DataFrame) else None
    rows.append({
        "Файл": f"{ICONS.get(ext, '📦')} {name}",
        "Тип": ext or "—",
        "Размер, КБ": round(os.path.getsize(p) / 1024, 1),
        "Строк": shape[0] if shape else (len(obj.splitlines()) if isinstance(obj, str) else len(obj)),
        "Колонок": shape[1] if shape else "—",
        "Изменён": datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%d.%m.%Y %H:%M"),
    })
inventory = pd.DataFrame(rows)
if not inventory.empty:
    display(inventory.style.bar(subset=["Размер, КБ"], color="#ff99ac")
            .set_properties(**{"text-align": "left"}).hide(axis="index"))
    total = sum(os.path.getsize(p) for p in paths)
    print(f"Всего занято: {human_size(total)}")


# ---------- 2. 🩺 Здоровье данных ----------
banner("Проверка здоровья таблиц", "🩺", "#11998e", "#38ef7d")
health = []
for name, df in TABLES.items():
    num = df.select_dtypes("number")
    missing = df.isna().mean().mean() * 100
    dups = df.duplicated().mean() * 100
    if not num.empty and not is_mnist(name, df):
        q1, q3 = num.quantile(.25), num.quantile(.75)
        iqr = q3 - q1
        outl = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).mean().mean() * 100
    else:
        outl = 0.0
    score = max(0, 100 - missing * 2 - dups * 2 - outl)
    health.append({"Таблица": name, "Пропуски, %": round(missing, 2),
                   "Дубликаты, %": round(dups, 2), "Выбросы (IQR), %": round(outl, 2),
                   "Оценка": round(score, 1),
                   "Статус": "💚 Отлично" if score > 90 else "💛 Норм" if score > 70 else "❤️‍🩹 Есть проблемы"})
if health:
    display(pd.DataFrame(health).style
            .background_gradient(subset=["Оценка"], cmap="RdYlGn", vmin=50, vmax=100)
            .hide(axis="index"))


# ---------- 3. 👀 Превью каждого файла ----------
banner("Быстрый взгляд на каждый файл", "👀", "#f7971e", "#ffd200")
for name, obj in DATA.items():
    display(Markdown(f"### {name}"))
    if isinstance(obj, pd.DataFrame):
        if is_mnist(name, obj):
            print(f"Изображения цифр 28×28: {len(obj)} шт. Метки: "
                  f"{dict(sorted(obj[0].value_counts().items()))}")
        else:
            display(obj.head())
            display(obj.describe().T.style.background_gradient(cmap="Blues", subset=["mean"]))
    elif isinstance(obj, (list, dict)):
        print(json.dumps(obj, ensure_ascii=False, indent=2)[:800])
    else:
        text = str(obj)
        print(text[:800] + ("…" if len(text) > 800 else ""))


# ---------- 4. 🔥 Корреляции ----------
for name, df in TABLES.items():
    num = df.select_dtypes("number")
    if is_mnist(name, df) or num.shape[1] < 2:
        continue
    banner(f"Корреляции: {name}", "🔥", "#eb3349", "#f45c43")
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(num.corr(), dtype=bool))
    sns.heatmap(num.corr(), mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=.5, square=True)
    plt.title(f"Матрица корреляций — {name}")
    plt.tight_layout(); plt.show()


# ---------- 5. 🗺 Карта Калифорнии ----------
for name, df in TABLES.items():
    if {"longitude", "latitude", "median_house_value"} <= set(df.columns):
        banner(f"Карта цен на жильё: {name}", "🗺", "#2193b0", "#6dd5ed")
        fig, ax = plt.subplots(figsize=(10, 8))
        size = df["population"] / 100 if "population" in df else 10
        sc = ax.scatter(df["longitude"], df["latitude"], s=size,
                        c=df["median_house_value"], cmap="turbo", alpha=.4)
        plt.colorbar(sc, label="Медианная стоимость дома, $")
        ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
        ax.set_title("Чем краснее — тем дороже (размер точки = население)")
        plt.show()
        top = df.nlargest(1, "median_house_value").iloc[0]
        print(f"💰 Самый дорогой район: ({top.latitude:.2f}, {top.longitude:.2f}) — "
              f"${top.median_house_value:,.0f}")
        break  # одной карты достаточно


# ---------- 6. ✍️ MNIST: галерея и «средняя цифра» ----------
for name, df in TABLES.items():
    if not is_mnist(name, df):
        continue
    banner(f"Рукописные цифры: {name}", "✍️", "#8E2DE2", "#4A00E0")
    labels, pixels = df.iloc[:, 0].values, df.iloc[:, 1:].values
    fig, axes = plt.subplots(3, 10, figsize=(14, 4.5))
    for ax, i in zip(axes.flat, random.sample(range(len(df)), 30)):
        ax.imshow(pixels[i].reshape(28, 28), cmap="magma")
        ax.set_title(labels[i], fontsize=12); ax.axis("off")
    plt.suptitle("30 случайных цифр"); plt.show()

    fig, axes = plt.subplots(1, 10, figsize=(14, 2))
    for d, ax in enumerate(axes):
        if (labels == d).any():
            ax.imshow(pixels[labels == d].mean(axis=0).reshape(28, 28), cmap="inferno")
        ax.set_title(f"«{d}»"); ax.axis("off")
    plt.suptitle("Средний портрет каждой цифры 👻"); plt.show()
    break


# ---------- 7. 🎭 Квартет Энскомба ----------
for name, df in TABLES.items():
    if {"Series", "X", "Y"} <= set(df.columns):
        banner("Квартет Энскомба — одинаковая статистика, разные данные", "🎭", "#fc4a1a", "#f7b733")
        series = sorted(df["Series"].unique())
        fig, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 4), sharey=True)
        for ax, s in zip(np.atleast_1d(axes), series):
            d = df[df["Series"] == s]
            k, b = np.polyfit(d["X"], d["Y"], 1)
            xs = np.linspace(d["X"].min(), d["X"].max(), 50)
            ax.scatter(d["X"], d["Y"], s=60, edgecolor="k")
            ax.plot(xs, k * xs + b, "r--")
            ax.set_title(f"Набор {s}")
            ax.text(.05, .95, f"μx={d.X.mean():.1f}  μy={d.Y.mean():.2f}\n"
                              f"r={d.X.corr(d.Y):.3f}\ny={k:.2f}x+{b:.2f}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=.8))
        plt.suptitle("Мораль: всегда смотри на график, а не только на цифры!")
        plt.tight_layout(); plt.show()


# ---------- 8. 🎲 Случайная запись дня ----------
banner("Случайная запись дня", "🎲", "#00b09b", "#96c93d")
candidates = {k: v for k, v in TABLES.items() if not is_mnist(k, v) and len(v)}
if candidates:
    tname = random.choice(list(candidates))
    print(f"Из файла {tname}:")
    display(candidates[tname].sample(1).T.rename(columns=lambda _: "значение"))


# ---------- 9. 🔎 Поиск по всем файлам ----------
def search(query, max_hits=20):
    """search('ocean') — ищет текст во всех файлах папки."""
    hits = 0
    q = query.lower()
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for n, line in enumerate(f, 1):
                if q in line.lower():
                    safe = html_lib.escape(line.strip()[:150])
                    i = safe.lower().find(html_lib.escape(q))
                    if i >= 0:
                        safe = (safe[:i] + "<mark>" + safe[i:i + len(html_lib.escape(q))]
                                + "</mark>" + safe[i + len(html_lib.escape(q)):])
                    display(HTML(f"<code>{os.path.basename(p)}:{n}</code> → {safe}"))
                    hits += 1
                    if hits >= max_hits:
                        print(f"…показаны первые {max_hits} совпадений"); return
    if hits == 0:
        print(f"Ничего не найдено по запросу «{query}» 🤷")


# ---------- 10. 🎛 Интерактивный исследователь ----------
banner("Интерактивный исследователь", "🎛", "#ee0979", "#ff6a00")
plain_tables = {k: v for k, v in TABLES.items() if not is_mnist(k, v)}
if HAS_WIDGETS and plain_tables:
    f_dd = widgets.Dropdown(options=list(plain_tables), description="Файл:")
    c_dd = widgets.Dropdown(description="Колонка:")
    k_dd = widgets.ToggleButtons(options=["Гистограмма", "Boxplot", "Топ значений"],
                                 description="График:")
    bins = widgets.IntSlider(value=30, min=5, max=100, description="Корзины:")
    q_txt = widgets.Text(placeholder="например: ocean", description="Поиск:")
    q_btn = widgets.Button(description="Найти", icon="search", button_style="info")
    q_out = widgets.Output()

    def update_cols(*_):
        c_dd.options = list(plain_tables[f_dd.value].columns)
    f_dd.observe(update_cols, "value"); update_cols()

    def draw(file, col, kind, bins):
        if col is None:
            return
        s = plain_tables[file][col].dropna()
        plt.figure(figsize=(9, 4))
        if kind == "Топ значений" or not pd.api.types.is_numeric_dtype(s):
            s.astype(str).value_counts().head(15).plot.barh(color="#ff6a88")
        elif kind == "Boxplot":
            sns.boxplot(x=s, color="#6dd5ed")
        else:
            sns.histplot(s, bins=bins, kde=True, color="#764ba2")
        plt.title(f"{col} · {file}"); plt.tight_layout(); plt.show()
        if pd.api.types.is_numeric_dtype(s):
            print(f"min={s.min():.3g}  max={s.max():.3g}  mean={s.mean():.3g}  "
                  f"median={s.median():.3g}  std={s.std():.3g}")

    def on_search(_):
        with q_out:
            q_out.clear_output(); search(q_txt.value)
    q_btn.on_click(on_search)

    out = widgets.interactive_output(draw, {"file": f_dd, "col": c_dd, "kind": k_dd, "bins": bins})
    display(widgets.VBox([widgets.HBox([f_dd, c_dd]), k_dd, bins, out,
                          widgets.HTML("<hr><b>🔎 Поиск по всем файлам</b>"),
                          widgets.HBox([q_txt, q_btn]), q_out]))
else:
    print("ipywidgets недоступен — используй search('текст') вручную.")


# ---------- 11. 📄 HTML-отчёт ----------
def make_report(path="/content/data_report.html", download=True):
    """make_report() — сохраняет красивый отчёт и скачивает его."""
    parts = [f"<h1>📊 Отчёт по {DATA_DIR}</h1><p>{datetime.datetime.now():%d.%m.%Y %H:%M}</p>",
             "<h2>Файлы</h2>", inventory.to_html(index=False)]
    if health:
        parts += ["<h2>Здоровье данных</h2>", pd.DataFrame(health).to_html(index=False)]
    for name, df in plain_tables.items():
        parts += [f"<h2>{name}</h2>", df.describe().round(3).to_html()]
    style = ("<style>body{font-family:sans-serif;margin:40px;color:#333}"
             "table{border-collapse:collapse;margin:10px 0}"
             "td,th{border:1px solid #ddd;padding:6px 10px}"
             "th{background:#764ba2;color:white}tr:nth-child(even){background:#f6f6fb}</style>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><meta charset='utf-8'>{style}</head><body>{''.join(parts)}</body></html>")
    print(f"✅ Отчёт сохранён: {path}")
    if download and IN_COLAB:
        colab_files.download(path)

banner("Готово! Дополнительные команды", "🏁")
display(Markdown("""
- `search("ocean")` — поиск текста во всех файлах
- `make_report()` — сохранить и скачать HTML-отчёт
- `DATA["имя_файла"]` — получить загруженный файл (таблицы — это pandas DataFrame)
"""))
