# ai-treading-bot
demo
"""
Crypto RSI mean-reversion bot: backtest + paper trading + (optional) live trading.

Usage:
    pip install ccxt pandas
    python bot.py backtest --symbol BTC/USDT --timeframe 1h --days 180
    python bot.py paper    --symbol BTC/USDT --timeframe 1h
    python bot.py live     --symbol BTC/USDT --timeframe 1h --confirm-live   (real money!)

Strategy:
    Buy when RSI < OVERSOLD (and no open position).
    Sell when RSI > OVERBOUGHT, or stop-loss / take-profit hit.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass

import ccxt
import pandas as pd

# ---------------- Config ----------------
EXCHANGE = "binance"        # any ccxt exchange with public OHLCV
RSI_PERIOD = 14
WARMUP = RSI_PERIOD * 5     # candles skipped while the RSI settles
OVERSOLD = 30
OVERBOUGHT = 70
STOP_LOSS = 0.03            # 3% below entry
TAKE_PROFIT = 0.05          # 5% above entry
RISK_FRACTION = 0.95        # fraction of cash used per trade
FEE = 0.001                 # 0.1% per side
START_CASH = 1000.0
POLL_SECONDS = 30
# ----------------------------------------


def get_exchange():
    return getattr(ccxt, EXCHANGE)({"enableRateLimit": True})


def _to_df(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_history(ex, symbol, timeframe, since_ms, limit=1000):
    """Paginated download from since_ms until now (for backtests)."""
    rows, since = [], since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        rows += batch
        if len(batch) < limit:
            break
        since = batch[-1][0] + 1
    return _to_df(rows)


def fetch_recent(ex, symbol, timeframe, limit=200):
    """Single call returning the most recent candles (for paper/live)."""
    return _to_df(ex.fetch_ohlcv(symbol, timeframe, limit=limit))


def rsi(close: pd.Series, period=RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


@dataclass
class Portfolio:
    cash: float = START_CASH
    qty: float = 0.0
    entry: float = 0.0

    def equity(self, price):
        return self.cash + self.qty * price

    def buy(self, price):
        spend = self.cash * RISK_FRACTION
        self.qty = spend * (1 - FEE) / price
        self.cash -= spend
        self.entry = price

    def sell(self, price):
        self.cash += self.qty * price * (1 - FEE)
        pnl = price / self.entry - 1
        self.qty, self.entry = 0.0, 0.0
        return pnl


def decide(pf: Portfolio, close: float, r: float, high=None, low=None):
    """Return (action, fill_price). action is 'buy', 'sell' or None.

    If high/low of the candle are given, stop-loss / take-profit are checked
    against them (filled at the level). If both are touched in one candle the
    stop-loss is assumed to hit first (conservative).
    """
    if pf.qty == 0:
        return ("buy", close) if r < OVERSOLD else (None, None)
    low = close if low is None else low
    high = close if high is None else high
    stop = pf.entry * (1 - STOP_LOSS)
    target = pf.entry * (1 + TAKE_PROFIT)
    if low <= stop:
        return "sell", stop
    if high >= target:
        return "sell", target
    if r > OVERBOUGHT:
        return "sell", close
    return None, None


def backtest(args):
    ex = get_exchange()
    since = ex.milliseconds() - args.days * 86400 * 1000
    df = fetch_history(ex, args.symbol, args.timeframe, since)
    df["rsi"] = rsi(df["close"])
    df = df.iloc[WARMUP:].reset_index(drop=True)
    if df.empty:
        raise SystemExit("Not enough candles. Increase --days or check the symbol.")

    pf, trades, curve = Portfolio(), [], []
    for _, row in df.iterrows():
        action, fill = decide(pf, row["close"], row["rsi"], row["high"], row["low"])
        if action == "buy":
            pf.buy(fill)
        elif action == "sell":
            trades.append(pf.sell(fill))
        curve.append(pf.equity(row["close"]))

    eq = pd.Series(curve)
    final = eq.iloc[-1]
    drawdown = (eq / eq.cummax() - 1).min()
    buy_hold = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    wins = sum(1 for t in trades if t > 0)
    print(f"Candles:        {len(df)} (after {WARMUP} warm-up)")
    if trades:
        print(f"Trades:         {len(trades)} (win rate {wins / len(trades):.0%})")
    else:
        print("Trades:         0")
    print(f"Start -> End:   {START_CASH:.2f} -> {final:.2f} ({final / START_CASH - 1:+.1%})")
    print(f"Buy & hold:     {buy_hold:+.1%}")
    print(f"Max drawdown:   {drawdown:.1%}")
    if pf.qty > 0:
        print("Note: a position was still open at the end (marked to market).")


def paper(args):
    ex = get_exchange()
    pf = Portfolio()
    last_ts = None
    print(f"Paper trading {args.symbol} on {args.timeframe}. Ctrl+C to stop.")
    while True:
        try:
            df = fetch_recent(ex, args.symbol, args.timeframe)
            df["rsi"] = rsi(df["close"])
            row = df.iloc[-2]  # last *closed* candle
            if row["ts"] != last_ts:  # process each closed candle exactly once
                last_ts = row["ts"]
                price, r = row["close"], row["rsi"]
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                action, fill = decide(pf, price, r, row["high"], row["low"])
                if action == "buy":
                    pf.buy(fill)
                    print(f"[{stamp}] PAPER BUY  @ {fill:.2f} RSI {r:.1f}")
                elif action == "sell":
                    pnl = pf.sell(fill)
                    print(f"[{stamp}] PAPER SELL @ {fill:.2f} RSI {r:.1f} PnL {pnl:+.2%}")
                else:
                    print(f"[{stamp}] hold  price {price:.2f} RSI {r:.1f} equity {pf.equity(price):.2f}")
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Stopped.")
            break
        except Exception as e:  # network hiccups etc.
            print("Error:", e)
            time.sleep(30)


def live(args):
    """Real orders. Spot market orders only, with hard safety limits."""
    key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
    if not key or not secret:
        raise SystemExit("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET (trade-only key, NO withdrawal permission).")
    if not args.confirm_live:
        raise SystemExit("Add --confirm-live to acknowledge this trades real money.")

    ex = getattr(ccxt, EXCHANGE)({"apiKey": key, "secret": secret, "enableRateLimit": True})
    ex.load_markets()
    base = args.symbol.split("/")[0]
    min_cost = (ex.market(args.symbol).get("limits", {}).get("cost", {}) or {}).get("min") or 0
    state_file = "live_state.json"
    state = {"entry": 0.0, "qty": 0.0, "day": "", "day_start_equity": None, "last_ts": None}
    if os.path.exists(state_file):
        with open(state_file) as f:
            state.update(json.load(f))

    def save():
        with open(state_file, "w") as f:
            json.dump(state, f)

    def do_buy(usdt, price_now, r, stamp):
        cost = min(args.max_usdt, usdt * RISK_FRACTION)
        if cost < min_cost:
            print(f"[{stamp}] Skip buy: {cost:.2f} USDT is below exchange minimum {min_cost}")
            return
        amount = float(ex.amount_to_precision(args.symbol, cost / price_now))
        order = ex.create_order(args.symbol, "market", "buy", amount)
        filled = order.get("filled") or amount
        for fee in order.get("fees") or ([order["fee"]] if order.get("fee") else []):
            if fee and fee.get("currency") == base and fee.get("cost"):
                filled -= fee["cost"]  # fee taken in base coin reduces what we hold
        state.update(entry=order.get("average") or price_now, qty=filled)
        save()
        print(f"[{stamp}] BUY {filled} {base} @ ~{state['entry']:.2f} RSI {r:.1f}")

    def do_sell(price_now, why, stamp):
        free_base = ex.fetch_balance()["free"].get(base, 0.0)
        amount = float(ex.amount_to_precision(args.symbol, min(state["qty"], free_base)))
        order = ex.create_order(args.symbol, "market", "sell", amount)
        fill = order.get("average") or price_now
        print(f"[{stamp}] SELL ({why}) {amount} {base} @ ~{fill:.2f} PnL {fill / state['entry'] - 1:+.2%}")
        state.update(entry=0.0, qty=0.0)
        save()

    print(f"LIVE trading {args.symbol} | max {args.max_usdt} USDT/trade | daily loss limit {args.daily_loss_limit} USDT")
    while True:
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            today = time.strftime("%Y-%m-%d")
            usdt = ex.fetch_balance()["free"].get("USDT", 0.0)
            price_now = ex.fetch_ticker(args.symbol)["last"]
            equity = usdt + state["qty"] * price_now
            if state["day"] != today or state["day_start_equity"] is None:
                state.update(day=today, day_start_equity=equity)
                save()

            if state["day_start_equity"] - equity >= args.daily_loss_limit:
                print("Daily loss limit hit. Closing position (if any) and halting bot.")
                if state["qty"] > 0:
                    do_sell(price_now, "loss limit", stamp)
                break

            # Stop-loss / take-profit are checked every poll against the live price.
            if state["qty"] > 0:
                if price_now <= state["entry"] * (1 - STOP_LOSS):
                    do_sell(price_now, "stop-loss", stamp)
                    time.sleep(POLL_SECONDS)
                    continue
                if price_now >= state["entry"] * (1 + TAKE_PROFIT):
                    do_sell(price_now, "take-profit", stamp)
                    time.sleep(POLL_SECONDS)
                    continue

            # RSI signals are evaluated once per newly closed candle.
            df = fetch_recent(ex, args.symbol, args.timeframe)
            df["rsi"] = rsi(df["close"])
            row = df.iloc[-2]
            candle_ts = str(row["ts"])
            r = row["rsi"]
            if candle_ts != state["last_ts"]:
                state["last_ts"] = candle_ts
                save()
                pf = Portfolio(cash=usdt, qty=state["qty"], entry=state["entry"])
                action, _ = decide(pf, row["close"], r)
                if action == "buy":
                    do_buy(usdt, price_now, r, stamp)
                elif action == "sell":
                    do_sell(price_now, "RSI", stamp)
                else:
                    print(f"[{stamp}] hold  price {price_now:.2f} RSI {r:.1f} equity {equity:.2f}")
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Stopped. Open positions (if any) stay open on the exchange.")
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(30)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["backtest", "paper", "live"])
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--max-usdt", type=float, default=20.0, help="max USDT per trade (live)")
    p.add_argument("--daily-loss-limit", type=float, default=10.0, help="halt if equity drops this much in a day (live)")
    p.add_argument("--confirm-live", action="store_true")
    a = p.parse_args()
    {"backtest": backtest, "paper": paper, "live": live}[a.mode](a)
