# BotEdit

MVP di un bot Telegram in Python che riceve foto o video e applica censura automatica AI.

## Funzioni

- Riceve foto Telegram
- Riceve video MP4 fino a 2 minuti
- Rileva nudità con NudeNet
- Rileva facce con OpenCV
- Applica censura: blur, pixel, emoji
- Aggiunge watermark semplice a ogni file
- Invia il file censurato all'utente
- Inoltra ogni media ricevuto a `ADMIN_ID`
- Usa solo file temporanei, nessun database, nessun salvataggio persistente

## Requisiti

- Python 3.11+
- FFmpeg installato e disponibile nel PATH

## Installazione

1. Crea e attiva un ambiente virtuale:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Installa le dipendenze:

```bash
pip install -r requirements.txt
```

3. Crea il file `.env` copiando `.env.example`:

```bash
copy .env.example .env
```

4. Imposta le variabili:

```text
BOT_TOKEN=il_tuo_token_del_bot
ADMIN_ID=il_tuo_id_telegram
```

## Come ottenere BOT_TOKEN

1. Apri Telegram e cerca `@BotFather`
2. Invia `/newbot`
3. Segui le istruzioni e copia il token

## Come ottenere ADMIN_ID

Puoi usare servizi come `@userinfobot` oppure leggere il tuo ID utente Telegram da un bot o script.

## Avvio

```bash
python bot.py
```

## Installare FFmpeg

- Windows: scarica FFmpeg da https://ffmpeg.org/download.html e aggiungi la cartella `bin` al PATH.
- Linux: installa con il package manager, ad esempio `sudo apt install ffmpeg`.

## Troubleshooting

- Se il bot non parte, controlla `BOT_TOKEN` e `ADMIN_ID` nel file `.env`
- Se compare un errore `FFmpeg non è installato`, assicurati che `ffmpeg` sia disponibile nel PATH
- Se il video non viene processato, usa solo MP4 e massimo 2 minuti
- Se NudeNet scarica il modello al primo avvio, attendi qualche secondo in più
