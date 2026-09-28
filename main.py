import os
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"
RESOLUTION = "15m"

# ===========================
# TELEGRAM SETTINGS
# ===========================

BOT_TOKEN = "8994783493:AAE5RV_3rTo3SxvUFMkXbH07s9Cm5fyxqzk"
CHAT_ID = "8555061802" 

# Last 24 hours = 96 x 15-minute candles
CANDLES_REQUIRED = 96

# Analysis refresh every 60 seconds
REFRESH_SECONDS = 10

# Topic 1: File Handling Path
LOG_FILE = "trading_signals_history.csv"


# ============================================================
# FETCH LAST 24 HOURS OF 15-MINUTE CANDLES
# ============================================================


def fetch_candles():
    end_time = int(time.time())
    start_time = end_time - (24 * 60 * 60)

    url = f"{BASE_URL}/v2/history/candles"

    params = {
        "symbol": SYMBOL,
        "resolution": RESOLUTION,
        "start": start_time,
        "end": end_time,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise Exception(f"API Error: {data}")

    candles = data["result"]

    df = pd.DataFrame(candles)

    required_columns = ["time", "open", "high", "low", "close", "volume"]

    for column in required_columns:
        if column not in df.columns:
            raise Exception(f"Missing column: {column}")

    df = df[required_columns].copy()

    # Convert values to numeric
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

#for data cleaning 

    df = (
        df.dropna()
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .tail(CANDLES_REQUIRED)
        .reset_index(drop=True)
    )

    if len(df) < 50:
        raise Exception(
            f"Not enough candles received. Received only {len(df)} candles."
        )

    return df


# ============================================================
# INDICATORS & ADVANCED METRICS
# ============================================================


def calculate_indicators(df):
    # EMA
    df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # ATR 14
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - previous_close)
    tr3 = abs(df["low"] - previous_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    # Volume average
    df["VOLUME_AVG"] = df["volume"].rolling(20).mean()

    # --- TOPIC 2: STATISTICAL ANALYSIS (Z-Score calculation) ---
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["VOL_ZSCORE"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["VOL_ZSCORE"] = df["VOL_ZSCORE"].fillna(0)

    df["RETURNS"] = df["close"].pct_change()

    return df


# ============================================================
# TOPIC 3: PROBABILITY & HYPOTHESIS TESTING
# ============================================================


def run_hypothesis_test(df):
    clean_returns = df["RETURNS"].dropna().values
    if len(clean_returns) < 10:
        return 1.0, False
    t_stat, p_value = stats.ttest_1samp(clean_returns, popmean=0.0)
    return p_value, p_value < 0.05


# ============================================================
# TOPIC 4: MACHINE LEARNING MODEL
# ============================================================


def train_ml_classifier(df):
    data = df.copy()
    data["TARGET"] = (data["close"].shift(-1) > data["open"].shift(-1)).astype(
        int
    )

    feature_cols = [
        "EMA_9",
        "EMA_21",
        "RSI",
        "MACD",
        "MACD_HIST",
        "ATR",
        "VOL_ZSCORE",
    ]
    clean_data = data.dropna(subset=feature_cols + ["TARGET"])

    if len(clean_data) < 30:
        return 0.5

    X = clean_data[feature_cols].iloc[:-1]
    y = clean_data["TARGET"].iloc[:-1]

    clf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
    clf.fit(X, y)

    current_features = clean_data[feature_cols].iloc[[-1]]
    prob_up = clf.predict_proba(current_features)[0][1]
    return prob_up


# ============================================================
# GENERATE BUY / SELL / HOLD SIGNAL
# ============================================================


def generate_signal(df):

    last = df.iloc[-1]
    previous = df.iloc[-2]

    price = float(last["close"])
    atr = float(last["ATR"])
    rsi = float(last["RSI"])

    buy_score = 0
    sell_score = 0

    buy_reasons = []
    sell_reasons = []

    # --------------------------------------------------------
    # EMA TREND
    # --------------------------------------------------------

    if last["EMA_9"] > last["EMA_21"]:
        buy_score += 2
        buy_reasons.append("EMA 9 is above EMA 21")
    else:
        sell_score += 2
        sell_reasons.append("EMA 9 is below EMA 21")

    if last["EMA_21"] > last["EMA_50"]:
        buy_score += 2
        buy_reasons.append("EMA 21 is above EMA 50")
    else:
        sell_score += 2
        sell_reasons.append("EMA 21 is below EMA 50")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if 52 <= rsi <= 70:
        buy_score += 2
        buy_reasons.append(f"RSI bullish ({rsi:.1f})")

    elif 30 <= rsi <= 48:
        sell_score += 2
        sell_reasons.append(f"RSI bearish ({rsi:.1f})")

    elif rsi > 75:
        sell_score += 1
        sell_reasons.append(f"RSI overbought ({rsi:.1f})")

    elif rsi < 25:
        buy_score += 1
        buy_reasons.append(f"RSI oversold ({rsi:.1f})")

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if last["MACD"] > last["MACD_SIGNAL"]:
        buy_score += 2
        buy_reasons.append("MACD bullish")

    else:
        sell_score += 2
        sell_reasons.append("MACD bearish")

    # MACD momentum
    if last["MACD_HIST"] > previous["MACD_HIST"]:
        buy_score += 1
    else:
        sell_score += 1

    # --------------------------------------------------------
    # CURRENT CANDLE
    # --------------------------------------------------------

    if last["close"] > last["open"]:
        buy_score += 1
        buy_reasons.append("Current candle bullish")
    else:
        sell_score += 1
        sell_reasons.append("Current candle bearish")

    # --------------------------------------------------------
    # 24-HOUR POSITION
    # --------------------------------------------------------

    high_24h = float(df["high"].max())
    low_24h = float(df["low"].min())

    range_24h = high_24h - low_24h

    if range_24h > 0:
        position_in_range = (price - low_24h) / range_24h

        if position_in_range > 0.60:
            buy_score += 1
            buy_reasons.append("Price in upper part of 24H range")

        elif position_in_range < 0.40:
            sell_score += 1
            sell_reasons.append("Price in lower part of 24H range")

    # --------------------------------------------------------
    # ADVANCED DS MODULES INTEGRATION
    # --------------------------------------------------------

    p_value, is_significant = run_hypothesis_test(df)
    if is_significant:
        buy_score += 1
        sell_score += 1
        buy_reasons.append(
            f"Statistically significant trend detected (p={p_value:.3f})"
        )

    ml_up_prob = train_ml_classifier(df)
    if ml_up_prob > 0.60:
        buy_score += 2
        buy_reasons.append(
            f"ML Model bullish prediction ({ml_up_prob*100:.1f}%)"
        )
    elif ml_up_prob < 0.40:
        sell_score += 2
        sell_reasons.append(
            f"ML Model bearish prediction ({(1-ml_up_prob)*100:.1f}%)"
        )

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    difference = abs(buy_score - sell_score)

    if buy_score >= 7 and buy_score >= sell_score + 3:

        signal = "BUY"

        stop_loss = price - (1.5 * atr)
        target_1 = price + (2.0 * atr)
        target_2 = price + (3.0 * atr)

        confidence = min(95, 50 + difference * 5)

    elif sell_score >= 7 and sell_score >= buy_score + 3:

        signal = "SELL"

        stop_loss = price + (1.5 * atr)
        target_1 = price - (2.0 * atr)
        target_2 = price - (3.0 * atr)

        confidence = min(95, 50 + difference * 5)

    else:

        signal = "HOLD"

        stop_loss = None
        target_1 = None
        target_2 = None

        confidence = max(buy_score, sell_score) * 8

    # --------------------------------------------------------
    # SIGNAL VALIDITY
    # --------------------------------------------------------

    if signal == "HOLD":
        validity_minutes = 15

    elif confidence >= 85:
        validity_minutes = 45

    elif confidence >= 70:
        validity_minutes = 30

    else:
        validity_minutes = 15

    return {
        "signal": signal,
        "price": price,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "confidence": confidence,
        "validity_minutes": validity_minutes,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "rsi": rsi,
        "atr": atr,
        "high_24h": high_24h,
        "low_24h": low_24h,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
    }


# ============================================================
# TOPIC 1: FILE HANDLING (PERSISTENCE)
# ============================================================


def log_signal_to_csv(result):
    log_data = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "signal": result["signal"],
        "price": result["price"],
        "confidence": result["confidence"],
        "rsi": result["rsi"],
        "atr": result["atr"],
    }

    df_log = pd.DataFrame([log_data])

    if not os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, index=False, mode="w")
    else:
        df_log.to_csv(LOG_FILE, index=False, mode="a", header=False)


# ============================================================
# TOPIC 5: DATA VISUALIZATION (SILENT CHART SAVER)
# ============================================================


def save_chart_image(df, result):
    try:
        # BTCfixed.py jis folder me hai
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Exact image path
        chart_path = os.path.join(base_dir, "latest_chart.png")

        plt.figure(figsize=(10, 5))

        plt.plot(
            df["time"],
            df["close"],
            label="Close Price",
            color="blue"
        )

        plt.plot(
            df["time"],
            df["EMA_9"],
            label="EMA 9",
            color="orange",
            linestyle="--"
        )

        plt.plot(
            df["time"],
            df["EMA_21"],
            label="EMA 21",
            color="green",
            linestyle="--"
        )

        plt.title(
            f"BTCUSD 15m Analysis - Signal: "
            f"{result['signal']} ({result['confidence']}%)"
        )

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save image
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Chart saved successfully: {chart_path}")

    except Exception as e:
        print(f"Chart save error: {e}")

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)
# ============================================================
# DISPLAY RESULT (EXACT ORIGINAL OUTPUT FORMAT)
# ============================================================


def display_signal(result, df):
    

    print("\n" + "=" * 65)
    print("             BTCUSD LIVE 15-MINUTE ANALYSIS")
    print("=" * 65)

    print(
        f"Time UTC         : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"Candles Analysed: {len(df)}")
    print("History         : Last 24 Hours")
    print("Timeframe       : 15 Minutes")

    print("-" * 65)

    print(f"LIVE PRICE      : ${result['price']:,.2f}")
    print(f"SIGNAL          : {result['signal']}")
    print(f"CONFIDENCE      : {result['confidence']}%")

    print("-" * 65)

    print(f"RSI             : {result['rsi']:.2f}")
    print(f"ATR             : ${result['atr']:,.2f}")

    print(f"24H HIGH        : ${result['high_24h']:,.2f}")
    print(f"24H LOW         : ${result['low_24h']:,.2f}")

    print(f"BUY SCORE       : {result['buy_score']}")
    print(f"SELL SCORE      : {result['sell_score']}")

    print("-" * 65)

    if result["signal"] != "HOLD":

        risk_percent = (
            abs(result["price"] - result["stop_loss"]) / result["price"]
        ) * 100

        print(f"ENTRY           : ${result['price']:,.2f}")
        print(f"STOP LOSS       : ${result['stop_loss']:,.2f}")
        print(f"TARGET 1        : ${result['target_1']:,.2f}")
        print(f"TARGET 2        : ${result['target_2']:,.2f}")

        print(f"PRICE RISK      : {risk_percent:.2f}%")

        print(
            "SIGNAL VALID FOR: NEXT "
            f"{result['validity_minutes']} MINUTES"
        )

        print(
            "IMPORTANT       : Validity time is NOT a guaranteed exit time."
        )

    else:

        print("NO TRADE        : Market setup is not strong enough.")
        print("CHECK AGAIN     : After next 15-minute candle.")

    print("-" * 65)

    if result["signal"] == "BUY":

        print("BUY REASONS:")

        for reason in result["buy_reasons"]:
            print(f"  + {reason}")

    elif result["signal"] == "SELL":

        print("SELL REASONS:")

        for reason in result["sell_reasons"]:
            print(f"  - {reason}")

    print("=" * 65)


# ============================================================
# MAIN LIVE LOOP
# ============================================================


def legacy_main():

    print("BTCUSD Signal Bot Starting...")
    print("Signal-only mode: NO automatic orders will be placed.")

    while True:

        try:

            df = fetch_candles()

            df = calculate_indicators(df)

            result = generate_signal(df)

            # Background DS Operations (Silent)
            log_signal_to_csv(result)
            save_chart_image(df, result)

            display_signal(result, df)


            if result["signal"] == "BUY":

                message = (
                    f"🟢 BTCUSD BUY SIGNAL\n\n"
                    f"💰 Entry: ${result['price']:.2f}\n"
                    f"🎯 TP1: ${result['target_1']:.2f}\n"
                    f"🎯 TP2: ${result['target_2']:.2f}\n"
                    f"🛑 SL: ${result['stop_loss']:.2f}\n"
                    f"📊 Confidence: {result['confidence']}%\n"
                    f"⏰ Valid: {result['validity_minutes']} Minutes"
                )

            elif result["signal"] == "SELL":

                message = (
                    f"🔴 BTCUSD SELL SIGNAL\n\n"
                    f"💰 Entry: ${result['price']:.2f}\n"
                    f"🎯 TP1: ${result['target_1']:.2f}\n"
                    f"🎯 TP2: ${result['target_2']:.2f}\n"
                    f"🛑 SL: ${result['stop_loss']:.2f}\n"
                    f"📊 Confidence: {result['confidence']}%\n"
                    f"⏰ Valid: {result['validity_minutes']} Minutes"
                )

            else:

                message = (
                    f"🟡 BTCUSD HOLD\n\n"
                    f"💰 Current Price: ${result['price']:.2f}\n"
                    f"📊 Confidence: {result['confidence']}%\n\n"
                    f"⚠️ Market setup is not strong enough.\n"
                    f"🚫 Do not enter any trade.\n"
                    f"⏳ Wait for the next 15-minute candle."
                )

            send_telegram(message)

        except KeyboardInterrupt:

            print("\nBot stopped by user.")
            break

        except Exception as error:

            print(f"\nERROR: {error}")

        print(
            "\nRefreshing analysis in "
            f"{REFRESH_SECONDS} seconds..."
        )

        time.sleep(REFRESH_SECONDS)

async def telegram_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_candles()
        df = calculate_indicators(df)
        result = generate_signal(df)

 # SAVE CHART
        save_chart_image(df, result)
        
        if result["signal"] == "BUY":
            message = (
                f"🟢 BTCUSD BUY SIGNAL\n\n"
                f"💰 Entry: ${result['price']:.2f}\n"
                f"🎯 TP1: ${result['target_1']:.2f}\n"
                f"🎯 TP2: ${result['target_2']:.2f}\n"
                f"🛑 SL: ${result['stop_loss']:.2f}\n"
                f"📊 Confidence: {result['confidence']}%"
            )

        elif result["signal"] == "SELL":
            message = (
                f"🔴 BTCUSD SELL SIGNAL\n\n"
                f"💰 Entry: ${result['price']:.2f}\n"
                f"🎯 TP1: ${result['target_1']:.2f}\n"
                f"🎯 TP2: ${result['target_2']:.2f}\n"
                f"🛑 SL: ${result['stop_loss']:.2f}\n"
                f"📊 Confidence: {result['confidence']}%"
            )

        else:
            message = (
                f"🟡 BTCUSD HOLD\n\n"
                f"💰 Current Price: ${result['price']:.2f}\n"
                f"📊 Confidence: {result['confidence']}%\n"
                f"⏳ Wait for next candle."
            )

        await update.message.reply_text(message)

    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", telegram_signal))
    app.add_handler(CommandHandler("status", telegram_signal))

    print("Telegram Bot Started...")
    app.run_polling()


if __name__ == "__main__":
    main()