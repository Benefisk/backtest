# requests загружает котировки по HTTP, pandas обрабатывает таблицы,
# matplotlib строит график капитала.
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt

# Параметры эксперимента: тикер, начало истории, длина SMA, комиссия
# и проскальзывание на каждую сторону сделки, безрисковая ставка
# и число торговых дней в году для годового коэффициента Sharpe.
TICKER = "SBER"
FROM_DATE = "2023-01-01"  # Начало загружаемой истории
SMA_PERIOD = 50
COMMISSION = 0.0005  # Комиссия на каждую сторону сделки: 0,05%
SLIPPAGE = 0.0003  # Проскальзывание на каждую сторону: 0,03%
RISK_FREE_RATE = 0.05  # Годовая безрисковая ставка для Sharpe
TRADING_DAYS_PER_YEAR = 252

# Кэш котировок в CSV: повторный запуск читает файл вместо сети,
# поэтому расчёт можно перезапустить с теми же данными.
# Удалите файл, чтобы скачать историю заново.
CACHE_FILE = Path(f"data/{TICKER}_daily.csv")


def download_prices():
    """Скачивает дневные свечи MOEX страницами и сохраняет их в CSV-кэш."""

    url = (
        "https://iss.moex.com/iss/engines/stock/"
        f"markets/shares/securities/{TICKER}/candles.json"
    )

    # Биржа отдаёт данные страницами; start указывает смещение следующей страницы.
    all_rows = []
    start = 0

    while True:
        response = requests.get(
            url,
            params={
                "from": FROM_DATE,
                "interval": 24,
                "start": start,
            },
            timeout=30,
        )
        # Прерываем работу с понятной HTTP-ошибкой (404, 500 и т.п.), если запрос не удался.
        response.raise_for_status()

        candles = response.json()["candles"]
        rows = [
            dict(zip(candles["columns"], row))
            for row in candles["data"]
        ]

        # Пустая страница означает, что котировки закончились.
        if not rows:
            break

        all_rows.extend(rows)
        start += len(rows)

        # Если страница неполная, следующей страницы уже нет.
        if len(rows) < 500:
            break

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise ValueError("Биржа не вернула данные.")

    # Приводим дату к типу datetime и используем её как индекс таблицы.
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.set_index("begin")

    # Сырые котировки сохраняем рядом с запуском для воспроизводимости.
    CACHE_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE_FILE)

    return df


def load_prices():
    """Берёт котировки из кэша; при первом запуске скачивает их с MOEX."""

    if CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    else:
        df = download_prices()

    df = df[["open", "close", "high", "low", "value", "volume"]].copy()

    # Убираем дни без торгов, пропуски и дубли; сортируем хронологически.
    df = df[df["volume"] > 0]
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if len(df) <= SMA_PERIOD:
        raise ValueError("Недостаточно данных для расчёта средней.")

    return df


def run_backtest(df):
    """Рассчитываем сигналы и условный результат."""

    df = df.copy()

    # SMA — простая скользящая средняя цены закрытия за выбранное число дней.
    df["sma"] = df["close"].rolling(SMA_PERIOD).mean()

    # Держим актив, когда закрытие выше SMA; иначе находимся в деньгах.
    df["signal"] = (df["close"] > df["sma"]).astype(int)

    # Используем сигнал предыдущего дня,
    # чтобы не применять будущее к прошлому
    df["position"] = df["signal"].shift(1).fillna(0)
    # Разница позиции между днями показывает вход (0→1) или выход (1→0).
    df["position_change"] = df["position"].diff().fillna(df["position"])
    # Считаем каждое открытие одной сделкой; последующий выход её закрывает.
    df["trade_count"] = (df["position_change"] > 0).astype(int)

    # Дневная доходность инструмента относительно предыдущего закрытия.
    df["market_return"] = df["close"].pct_change()

    # Доходность позиции за день минус комиссия за вход/выход.
    df["trades"] = df["position_change"].abs()

    df["strategy_return"] = (
        df["position"] * df["market_return"]
        - df["trades"] * (COMMISSION + SLIPPAGE)
    )

    # Последовательно перемножаем дневные доходности, начиная с капитала 1.
    df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df


def build_trade_log(df):
    """Журнал сделок: даты входа и выхода и результат каждой сделки."""

    log = []
    entry_price = None
    entry_date = None

    for date, row in df.iterrows():
        if row["position_change"] == 1:
            entry_price = row["close"]
            entry_date = date
        elif row["position_change"] == -1 and entry_price is not None:
            # Из закрытой сделки вычитаем издержки за вход и за выход.
            result = row["close"] / entry_price - 1 - 2 * (COMMISSION + SLIPPAGE)
            log.append(
                {"Вход": entry_date.date(), "Выход": date.date(),
                 "Результат, %": result * 100}
            )
            entry_price = None

    # Позиция, открытая на последний день, оценивается по последней цене;
    # издержки списаны только за вход.
    if entry_price is not None:
        result = df["close"].iloc[-1] / entry_price - 1 - (COMMISSION + SLIPPAGE)
        log.append(
            {"Вход": entry_date.date(), "Выход": None,
             "Результат, %": result * 100}
        )

    return pd.DataFrame(log)


def show_results(df):
    """Показываем метрики в отдельной панели рядом с графиком."""

    # Итоговый прирост капитала в процентах от начального значения 1.
    total_return = (df["equity"].iloc[-1] - 1) * 100

    # Просадка — падение от предыдущего максимума капитала; берём худшее.
    running_peak = df["equity"].cummax()
    drawdown = df["equity"] / running_peak - 1
    max_drawdown = drawdown.min() * 100

    # Sharpe: средняя дневная доходность / её стандартное отклонение,
    # умноженное на корень из 252 для пересчёта к годовому масштабу.
    daily_returns = df["strategy_return"].dropna()
    volatility = daily_returns.std(ddof=1)
    sharpe = (
        (daily_returns.mean() - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        / volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
        if len(daily_returns) > 1 and volatility > 0
        else 0.0
    )
    trade_count = int(df["trade_count"].sum())

    # Годовая доходность и дополнительные метрики качества.
    calendar_days = (df.index[-1] - df.index[0]).days
    annual_return = (
        (1 + total_return / 100) ** (365 / calendar_days) - 1
        if calendar_days > 0 and total_return > -100
        else 0.0
    )
    calmar = annual_return / abs(drawdown.min()) if drawdown.min() < 0 else 0.0

    # Ориентир «купи и держи» на тех же датах и ценах.
    buy_hold = df["close"] / df["close"].iloc[0]
    bh_return = (buy_hold.iloc[-1] - 1) * 100
    bh_drawdown = (buy_hold / buy_hold.cummax() - 1).min() * 100

    # Сводка по сделкам из журнала: доля прибыльных и profit factor.
    log = build_trade_log(df)
    if log.empty:
        wins_count, win_rate, profit_factor = 0, 0.0, 0.0
    else:
        wins = log["Результат, %"] > 0
        wins_count = int(wins.sum())
        win_rate = wins.mean() * 100
        gross_profit = log.loc[wins, "Результат, %"].sum()
        gross_loss = log.loc[~wins, "Результат, %"].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf")

    # Журнал печатаем в консоль: на графике он не поместится.
    print("\nЖурнал сделок:")
    print(log.to_string(index=False))

    # Создаём две панели: кривая капитала и сводка результатов.
    figure, (ax_chart, ax_stats) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [5, 1]}
    )
    ax_chart.plot(df.index, df["equity"], label="Стратегия")
    ax_chart.plot(df.index, buy_hold, label="В долгую", alpha=0.65)
    ax_chart.set_title("Бэктест стратегии SMA")
    ax_chart.set_xlabel("Дата")
    ax_chart.set_ylabel("Условный капитал")
    ax_chart.legend()
    ax_chart.grid(True)

    # Текстовая сводка размещается непосредственно в правой панели окна.
    ax_stats.axis("off")
    stats = (
        f"Инструмент: {TICKER}\n\n"
        f"Дневных свечей: {len(df)}\n"
        f"Начало периода: {df.index[0].date()}\n"
        f"Конец периода: {df.index[-1].date()}\n\n"
        f"Доходность: {total_return:.2f}%\n"
        f"Годовая доходность: {annual_return * 100:.2f}%\n"
        f"Sharpe годовой: {sharpe:.2f}\n"
        f"Макс. просадка: {max_drawdown:.2f}%\n"
        f"Calmar: {calmar:.2f}\n\n"
        f"Сделок (входов): {trade_count}\n"
        f"Прибыльных: {wins_count} ({win_rate:.0f}%)\n"
        f"Profit factor: {profit_factor:.2f}\n\n"
        f"В долгую: доходность {bh_return:.2f}%\n"
        f"В долгую: просадка {bh_drawdown:.2f}%\n\n"
        f"Комиссия: {COMMISSION * 100:.3f}% \n(+ проскальзывание "
        f"{SLIPPAGE * 100:.3f}% на сторону)"
    )
    ax_stats.text(
        0.03, 0.97, stats, transform=ax_stats.transAxes,
        va="top", ha="left", fontsize=11, linespacing=1.5
    )

    figure.tight_layout()
    figure.savefig("equity_curve.png")
    plt.show()


def main():
    # Полный цикл: загрузка котировок → бэктест → метрики и график.
    prices = load_prices()
    results = run_backtest(prices)
    show_results(results)


if __name__ == "__main__":
    main()
