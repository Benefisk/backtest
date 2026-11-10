# Библиотека для загрузки котировок с мос.биржи
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Настройки учебного эксперимента
TICKER = "SBER"
SMA_PERIOD = 50
COMMISSION = 0.0005  # 0,05% при изменении позиции
TRADING_DAYS_PER_YEAR = 252


def load_prices():
    """Загружаем дневные цены с Московской биржи."""

    url = (
        "https://iss.moex.com/iss/engines/stock/"
        f"markets/shares/securities/{TICKER}/candles.json"
    )

    all_rows = []
    start = 0

    while True:
        response = requests.get(
            url,
            params={
                "from": "2023-01-01",
                "interval": 24,
                "start": start,
            },
            timeout=30,
        )
        response.raise_for_status()

        candles = response.json()["candles"]
        rows = [
            dict(zip(candles["columns"], row))
            for row in candles["data"]
        ]

        if not rows:
            break

        all_rows.extend(rows)
        start += len(rows)

        if len(rows) < 500:
            break

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise ValueError("Биржа не вернула данные.")

    df["begin"] = pd.to_datetime(df["begin"])
    df = df.set_index("begin")
    df = df[["open", "close", "volume"]].copy()

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

    # Средняя цена закрытия за 50 торговых дней
    df["sma"] = df["close"].rolling(SMA_PERIOD).mean()

    # Сигнал: 1 — цена выше средней, 0 — иначе
    df["signal"] = (df["close"] > df["sma"]).astype(int)

    # Используем сигнал предыдущего дня,
    # чтобы не применять будущее к прошлому
    df["position"] = df["signal"].shift(1).fillna(0)
    df["position_change"] = df["position"].diff().fillna(df["position"])
    # Сделкой считаем вход в позицию; выход — закрытие этой сделки.
    df["trade_count"] = (df["position_change"] > 0).astype(int)

    # Изменение цены за день
    df["market_return"] = df["close"].pct_change()

    # Условная доходность стратегии
    df["trades"] = df["position_change"].abs()

    df["strategy_return"] = (
        df["position"] * df["market_return"]
        - df["trades"] * COMMISSION
    )

    # Рост условного капитала
    df["equity"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df


def show_results(df):
    """Печатаем статистику и строим график."""

    total_return = (df["equity"].iloc[-1] - 1) * 100

    running_peak = df["equity"].cummax()
    drawdown = df["equity"] / running_peak - 1
    max_drawdown = drawdown.min() * 100

    daily_returns = df["strategy_return"].dropna()
    volatility = daily_returns.std(ddof=1)
    sharpe = (
        daily_returns.mean() / volatility * (TRADING_DAYS_PER_YEAR ** 0.5)
        if len(daily_returns) > 1 and volatility > 0
        else 0.0
    )
    trade_count = int(df["trade_count"].sum())

    print(f"Инструмент: {TICKER}")
    print(f"Количество дневных свечей: {len(df)}")
    print(f"Начало периода: {df.index[0].date()}")
    print(f"Конец периода: {df.index[-1].date()}")
    print(f"Итоговая доходность: {total_return:.2f}%")
    print(f"Sharpe (годовой, без безрисковой ставки): {sharpe:.2f}")
    print(f"Максимальная просадка: {max_drawdown:.2f}%")
    print(f"Количество сделок (входов в позицию): {trade_count}")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["equity"], label="Стратегия")
    plt.title("Бэктест стратегии SMA")
    plt.xlabel("Дата")
    plt.ylabel("Условный капитал")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()


def main():
    prices = load_prices()
    results = run_backtest(prices)
    show_results(results)


if __name__ == "__main__":
    main()
