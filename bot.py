import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
from PIL import Image

try:
    from nudenet import NudeDetector
except ImportError:
    logger = logging.getLogger(__name__)
    logger.critical("NudeNet non è installato. Installa con: pip install nudenet")
    sys.exit(1)

try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

load_dotenv()

def check_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger = logging.getLogger(__name__)
        logger.critical(
            "FFmpeg non è installato o non è nel PATH. "
            "Installalo: Windows (scarica da https://ffmpeg.org), "
            "Linux (apt install ffmpeg), macOS (brew install ffmpeg)"
        )
        sys.exit(1)

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")

if not BOT_TOKEN or not ADMIN_ID:
    raise SystemExit("Missing BOT_TOKEN or ADMIN_ID in environment")

try:
    ADMIN_ID = int(ADMIN_ID)
except ValueError:
    raise SystemExit("ADMIN_ID must be an integer")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

check_ffmpeg()

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# In-memory session state for a single active upload per user.
sessions: dict[int, dict] = {}
video_processing_lock = asyncio.Lock()

# Face detector for OpenCV.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Nude detection model (lazy load).
try:
    detector = NudeDetector()
except Exception as exc:
    logger.warning("Impossibile caricare NudeNet, il modello sarà scaricato al primo utilizzo: %s", exc)
    detector = None

# Modern inline keyboard with emoji styling
INLINE_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="🌫️  Blur", callback_data="censor:blur"),
        InlineKeyboardButton(text="🟪  Pixel", callback_data="censor:pixel"),
        InlineKeyboardButton(text="😳  Emoji", callback_data="censor:emoji"),
    ],
])

BACK_BUTTON = InlineKeyboardButton(
    text="⬅️ Indietro",
    callback_data="back"
)

# Strength selector keyboard
STRENGTH_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="🟢 Light", callback_data="strength:light"),
        InlineKeyboardButton(text="🟠 Medium", callback_data="strength:medium"),
        InlineKeyboardButton(text="🔴 Strong", callback_data="strength:strong"),
    ],
    [
        BACK_BUTTON,
    ],
])

# Emoji selector keyboard
EMOJI_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="😳 Classic", callback_data="emoji:shocked"),
        InlineKeyboardButton(text="🍑 Peach", callback_data="emoji:peach"),
    ],
    [
        InlineKeyboardButton(text="🔞 NSFW", callback_data="emoji:18"),
        InlineKeyboardButton(text="💋 Lips", callback_data="emoji:lips"),
    ],
    [
        InlineKeyboardButton(text="🖤 Black", callback_data="emoji:black"),
    ],
    [
        BACK_BUTTON,
    ]
])

TARGET_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [
        InlineKeyboardButton(text="🔞 Nudità", callback_data="target:nude"),
        InlineKeyboardButton(text="👤 Volto", callback_data="target:face"),
    ],
    [
        InlineKeyboardButton(text="🛡️ Entrambi", callback_data="target:all"),
    ],
    [
        BACK_BUTTON,
    ]
])



# Emoji mapping
EMOJI_MAP = {
    "shocked": "😳",
    "peach": "🍑",
    "18": "🔞",
    "lips": "💋",
    "black": "🖤",
}


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def detect_regions(image_path: str, target: str = "all") -> list[tuple[int, int, int, int]]:
    image = cv2.imread(image_path)
    if image is None:
        return []

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes: list[tuple[int, int, int, int]] = []

    NUDE_CLASSES = {
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "BUTTOCKS_EXPOSED",
    }

    # =========================
    # NUDE DETECTION
    # =========================
    if target in ["nude", "all"] and detector is not None:
        try:
            results = detector.detect(image_path)
        except Exception as exc:
            logger.debug("NudeNet detection failed: %s", exc)
            results = []

        for result in results:
            cls = result.get("class")

            if cls not in NUDE_CLASSES:
                continue

            box = result.get("box")
            if not box or len(box) != 4:
                continue

            x, y, w, h = map(int, box)

            x1, y1, x2, y2 = clamp_box(
                x,
                y,
                x + w,
                y + h,
                width,
                height,
            )

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    # =========================
    # FACE DETECTION
    # =========================
    if target in ["face", "all"]:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for x, y, w, h in faces:
            x1, y1, x2, y2 = clamp_box(
                x,
                y,
                x + w,
                y + h,
                width,
                height,
            )

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def apply_censor(image: np.ndarray, box: tuple[int, int, int, int], style: str, strength: str = "medium", emoji: str = "shocked") -> None:
    x1, y1, x2, y2 = box
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return

    if style == "blur":
        # Blur intensity based on strength
        blur_kernels = {
            "light": (21, 21),
            "medium": (51, 51),
            "strong": (99, 99),
        }
        kernel = blur_kernels.get(strength, (51, 51))
        blur = cv2.GaussianBlur(region, kernel, 0)
        image[y1:y2, x1:x2] = blur
        
    elif style == "pixel":
        # Pixelation intensity based on strength
        height, width = region.shape[:2]
        if height < 8 or width < 8:
            image[y1:y2, x1:x2] = (0, 0, 0)
            return
        
        pixel_scales = {
            "light": 5,      # larger pixels
            "medium": 10,     # medium pixels
            "strong": 20,      # smaller, denser pixels
        }
        scale = pixel_scales.get(strength, 10)
        small = cv2.resize(region, (max(1, width // scale), max(1, height // scale)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = pixelated
        
    else:  # emoji style
        
        # Add emoji
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Emoji image path
        emoji_path = os.path.join("assets", f"{emoji}.png")

        # Calculate emoji size
        box_width = x2 - x1
        box_height = y2 - y1

        size_multipliers = {
            "light": 0.4,
            "medium": 0.7,
            "strong": 1.0,
        }

        size_mult = size_multipliers.get(strength, 0.7)
        emoji_size = int(min(box_width, box_height) * size_mult)
        emoji_size = max(40, emoji_size)

        try:
            # Load PNG emoji
            emoji_img = Image.open(emoji_path).convert("RGBA")

            # Resize emoji
            emoji_img = emoji_img.resize((emoji_size, emoji_size))

            # Center position
            tx = x1 + max(0, (box_width - emoji_size) // 2)
            ty = y1 + max(0, (box_height - emoji_size) // 2)

            # Paste PNG
            pil_img.paste(emoji_img, (tx, ty), emoji_img)

        except Exception as e:
            logger.debug("Emoji PNG error: %s", e)

        # Convert back to OpenCV
        image[:, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def apply_watermark(
    image: np.ndarray,
    text: str,
    position: str = "bottom",
    size: str = "medium",
    opacity: float = 0.5,
) -> None:

    if not text:
        return

    overlay = np.zeros_like(image)
    height, width = image.shape[:2]

    scales = {
        "small": 0.8,
        "medium": 1.5,
        "large": 2.5,
    }

    scale = scales.get(size, 1.5)

    thickness = max(1, int(scale * 2))

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )[0]

    text_width, text_height = text_size

    if position == "center":
        x = (width - text_width) // 2
        y = height // 2

    elif position == "diagonal":

        temp = np.zeros_like(image)

        text_x = (width - text_width) // 2
        text_y = height // 2

        cv2.putText(
            temp,
            text,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        matrix = cv2.getRotationMatrix2D(
            (width // 2, height // 2),
            -30,
            1,
        )

        rotated = cv2.warpAffine(
            temp,
            matrix,
            (width, height),
        )

        mask = rotated.astype(bool)

        blended = cv2.addWeighted(
            image,
            1.0,
            rotated,
            opacity,
            0,
        )

        image[mask] = blended[mask]

        return

    elif position == "bottom":
        x = 20
        y = height - 30

    else:
        x = 20
        y = 50

    cv2.putText(
        overlay,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    mask = overlay.astype(bool)

    blended = cv2.addWeighted(
        image,
        1.0,
        overlay,
        opacity,
        0,
    )

    image[mask] = blended[mask]


WATERMARK_POSITION_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🎯 Centro", callback_data="wmpos:center"),
            InlineKeyboardButton(text="📐 Diagonale", callback_data="wmpos:diagonal"),
        ],
        [
            InlineKeyboardButton(text="⬇️ Basso", callback_data="wmpos:bottom"),
            InlineKeyboardButton(text="↖️ Alto", callback_data="wmpos:top"),
        ],
        [
            BACK_BUTTON,
        ]
    ]
)

WATERMARK_SIZE_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🟢 Small", callback_data="wmsize:small"),
            InlineKeyboardButton(text="🟠 Medium", callback_data="wmsize:medium"),
            InlineKeyboardButton(text="🔴 Large", callback_data="wmsize:large"),
        ],
        [
            BACK_BUTTON,
        ]
    ]
)

WATERMARK_OPACITY_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="10%", callback_data="wmopacity:0.1"),
            InlineKeyboardButton(text="30%", callback_data="wmopacity:0.3"),
        ],
        [
            InlineKeyboardButton(text="60%", callback_data="wmopacity:0.6"),
            InlineKeyboardButton(text="90%", callback_data="wmopacity:0.9"),
        ],
        [
            BACK_BUTTON,
        ]
    ]
)

WATERMARK_ENABLE_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Sì", callback_data="wmenable:yes"),
            InlineKeyboardButton(text="❌ No", callback_data="wmenable:no"),
        ],
        [
            BACK_BUTTON
        ]
    ]
)

def process_image( input_path,
    output_path,
    style,
    strength="medium",
    emoji="shocked",
    target="all",
    watermark_text=None,
    watermark_position="bottom",
    watermark_size="medium",
    watermark_opacity=0.5,
    watermark_enabled=True
    ) -> None:
    
    image = cv2.imread(input_path)
    if image is None:
        raise RuntimeError("Impossibile aprire l'immagine")

    boxes = detect_regions(input_path, target)
    for box in boxes:
        apply_censor(image, box, style, strength, emoji)

    if watermark_enabled:
        apply_watermark(
            image,
            watermark_text,
            watermark_position,
            watermark_size,
            watermark_opacity,
        )
    success = cv2.imwrite(output_path, image)
    if not success:
        raise RuntimeError("Errore salvataggio immagine processata")


def get_video_metadata(video_path: str) -> tuple[float, float]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Impossibile analizzare il video")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = frame_count / fps if fps else 0
    return fps, duration


def run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg non è installato o non è nel PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"FFmpeg failed: {exc.stderr.strip()}") from exc


def process_video(input_path,
    output_path,
    style,
    strength="medium",
    emoji="shocked",
    target="all",
    watermark_text=None,
    watermark_position="bottom",
    watermark_size="medium",
    watermark_opacity=0.5,
    watermark_enabled=True
    ) -> None:
    fps, duration = get_video_metadata(input_path)
    if duration > 120:
        raise RuntimeError("Il video supera il limite massimo di 2 minuti")

    frames_dir = tempfile.mkdtemp(prefix="botedit_frames_")
    try:
        frame_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-vf",
            "fps=12",
            "-q:v",
            "2",
            frame_pattern,
        ])

        frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
        if not frame_files:
            raise RuntimeError("Nessun frame estratto dal video")

        last_boxes = []

        for i, frame_file in enumerate(frame_files):

            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue

            # AI detection solo ogni 5 frame
            if i % 5 == 0:
                last_boxes = detect_regions(str(frame_file), target)

            for box in last_boxes:
                apply_censor(frame, box, style, strength, emoji)

            if watermark_enabled:    
                apply_watermark( frame,
                    watermark_text,
                    watermark_position,
                    watermark_size,
                    watermark_opacity,
                )
            cv2.imwrite(str(frame_file), frame)

        temp_video = os.path.join(frames_dir, "temp_video.mp4")

        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "12",
            "-i",
            os.path.join(frames_dir, "frame_%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            temp_video,
        ])

        run_ffmpeg([
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            temp_video,
            "-i",
            input_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:a",
            "aac",
            output_path,
        ])
        
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)



async def notify_admin(message: types.Message) -> None:
    text = f"Nuovo media da @{message.from_user.username or message.from_user.id}"
    try:
        await bot.send_message(ADMIN_ID, text)
        await bot.copy_message(chat_id=ADMIN_ID, from_chat_id=message.chat.id, message_id=message.message_id)
    except Exception as exc:
        logger.warning("Impossibile notificare l'admin: %s", exc)


def cleanup_session(user_id: int) -> None:
    session = sessions.pop(user_id, None)
    if not session:
        return
    temp_dir = session.get("temp_dir")
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


@dp.message(Command(commands=["start", "help"]))
async def start_command(message: types.Message) -> None:
    start_text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🛡️  <b>CoverIA Protect</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "<b>Censura automaticamente:</b>\n"
        "👤 Volti\n"
        "🔞 Nudità\n"
        "🎥 Video\n"
        "📸 Foto\n\n"
        "<b>✨ Modalità disponibili:</b>\n"
        "🌫️  Blur - sfocatura soft\n"
        "🟪 Pixel - pixelazione\n"
        "😳 Emoji - cover con emoji\n\n"
        "📤 Invia una foto o video per iniziare la censura AI\n"
        "(Max 2 minuti, formato MP4)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    await message.answer(
        start_text,
        parse_mode="HTML",
    )


@dp.message(F.photo | F.video)
async def media_handler(message: types.Message) -> None:
    user_id = message.from_user.id
    cleanup_session(user_id)

    if message.photo:
        media_type = "photo"
        extension = "jpg"
        file_id = message.photo[-1].file_id
        media_emoji = "📸"
    else:
        media_type = "video"
        extension = "mp4"
        file_id = message.video.file_id
        media_emoji = "🎥"
        
        if message.video.duration and message.video.duration > 120:
            error_text = (
                "❌ <b>Video troppo lungo</b>\n\n"
                "Carica un video con durata inferiore a 2 minuti."
            )
            await message.reply(error_text, parse_mode="HTML")
            return
        mime = message.video.mime_type or ""
        if "mp4" not in mime.lower() and not (message.video.file_name or "").lower().endswith(".mp4"):
            error_text = (
                "❌ <b>Formato non supportato</b>\n\n"
                "Usa solo video in formato <code>MP4</code>."
            )
            await message.reply(error_text, parse_mode="HTML")
            return

    # Show receiving message
    status_msg = await message.reply(
        f"✅ {media_emoji} Media ricevuto\n⏳ Analizzando...",
        parse_mode="HTML"
    )

    temp_dir = tempfile.mkdtemp(prefix="botedit_")
    file_path = os.path.join(temp_dir, f"input.{extension}")

    try:
        telegram_file = await bot.get_file(file_id)
        await bot.download(telegram_file, destination=file_path)
    except Exception as exc:
        cleanup_session(user_id)
        logger.error("Errore download file: %s", exc)
        error_text = (
            "❌ <b>Errore nel download</b>\n\n"
            "Riprova a inviare il media."
        )
        await status_msg.edit_text(error_text, parse_mode="HTML")
        return

    sessions[user_id] = {
        "media_path": file_path,
        "media_type": media_type,
        "temp_dir": temp_dir,
        "status_msg_id": status_msg.message_id,
    }

    # Notify admin silently
    await notify_admin(message)
    
    # Edit message to show options
    keyboard_text = (
        f"✅ {media_emoji} Media pronto\n\n"
        "🎨 <b>Scegli la modalità di censura:</b>"
    )
    await status_msg.edit_text(keyboard_text, reply_markup=INLINE_KEYBOARD, parse_mode="HTML")

@dp.callback_query(F.data.startswith("wmenable:"))
async def watermark_enable_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return

    enabled = callback.data.split(":", 1)[1] == "yes"

    sessions[user_id]["watermark_enabled"] = enabled

    await callback.answer("✨ Impostazione salvata")

    if not enabled:
        sessions[user_id]["watermark_enabled"] = False

        style = session.get("style")
        strength = session.get("strength", "medium")

        await process_media_async(
            callback,
            user_id,
            session,
            style,
            strength,
        )
        return

    # watermark enabled
    sessions[user_id]["watermark_enabled"] = True
    sessions[user_id]["awaiting_watermark_text"] = True

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=session["status_msg_id"],
        text=(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "✍️ <b>Scrivi la tua firma</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        ),
        parse_mode="HTML",
    )



@dp.callback_query(F.data.startswith("censor:"))
async def censor_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return

    style = callback.data.split(":", 1)[1]

    sessions[user_id]["style"] = style

    style_names = {
        "blur": "🌫️ Blur",
        "pixel": "🟪 Pixel",
        "emoji": "😳 Emoji",
    }

    await callback.answer(
        f"✨ {style_names.get(style, style)} selezionato"
    )

    target_text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🛡️ <b>Cosa vuoi censurare?</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🔞 Solo nudità\n"
        "👤 Solo volto\n"
        "🛡️ Entrambi"
    )

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=session["status_msg_id"],
        text=target_text,
        reply_markup=TARGET_KEYBOARD,
        parse_mode="HTML",
    )

@dp.callback_query(F.data.startswith("target:"))
async def target_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return

    target = callback.data.split(":", 1)[1]

    sessions[user_id]["target"] = target

    names = {
        "nude": "🔞 Nudità",
        "face": "👤 Volto",
        "all": "🛡️ Entrambi",
    }

    await callback.answer(
        f"✨ {names.get(target)} selezionato"
    )

    intensity_text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🎚️ <b>Seleziona intensità</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🟢 Light\n"
        "🟠 Medium\n"
        "🔴 Strong"
    )

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=session["status_msg_id"],
        text=intensity_text,
        reply_markup=STRENGTH_KEYBOARD,
        parse_mode="HTML",
    )
    
@dp.callback_query(F.data.startswith("strength:"))
async def strength_callback(callback: types.CallbackQuery) -> None:
    user_id = callback.from_user.id
    session = sessions.get(user_id)
    
    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return
    
    strength = callback.data.split(":", 1)[1]
    strength_names = {
        "light": "🟢 Light",
        "medium": "🟠 Medium",
        "strong": "🔴 Strong",
    }
    strength_name = strength_names.get(strength, strength)
    
    # Save strength to session
    sessions[user_id]["strength"] = strength
    
    await callback.answer(f"✨ {strength_name} selezionato")
    
    style = session.get("style")
    status_msg_id = session.get("status_msg_id")
    
    # If emoji mode, show emoji selector
    if style == "emoji":
        emoji_text = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "😳 <b>Seleziona emoji</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>Scegli l'emoji da usare:</b>"
        )
        
        if status_msg_id:
            try:
                await bot.edit_message_text(
                    chat_id=user_id,
                    message_id=status_msg_id,
                    text=emoji_text,
                    reply_markup=EMOJI_KEYBOARD,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.debug("Cannot edit message: %s", e)
        else:
            await callback.message.answer(emoji_text, reply_markup=EMOJI_KEYBOARD, parse_mode="HTML")
    else:

        text = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🛡️ <b>Vuoi aggiungere una firma?</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        await bot.edit_message_text(
            chat_id=user_id,
            message_id=session["status_msg_id"],
            text=text,
            reply_markup=WATERMARK_ENABLE_KEYBOARD,
            parse_mode="HTML",
        )
    
# @dp.message(Command("watermark"))
@dp.message()
async def watermark_text_handler(message: types.Message):

    user_id = message.from_user.id
    session = sessions.get(user_id)

    if not session:
        return

    if not session.get("awaiting_watermark_text"):
        return

    text = (message.text or "").strip()

    if not text:
        await message.answer("❌ Firma non valida")
        return

    if len(text) > 30:
        await message.answer(
            "❌ Massimo 30 caratteri"
        )
        return

    sessions[user_id]["custom_watermark"] = text
    sessions[user_id]["awaiting_watermark_text"] = False

    await message.answer(
        f"✅ Firma impostata:\n\n"
        f"<code>{text}</code>",
        parse_mode="HTML",
    )

    session = sessions[user_id]

    await bot.edit_message_text(
         chat_id=user_id,
        message_id=session["status_msg_id"],
        text="🛡️ Seleziona posizione watermark",
        reply_markup=WATERMARK_POSITION_KEYBOARD,
    )


@dp.message(Command("watermark"))
async def set_watermark(message: types.Message):
    user_id = message.from_user.id

    parts = message.text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            "❌ Usa:\n\n"
            "<code>/watermark TESTO</code>",
            parse_mode="HTML",
        )
        return

    text = parts[1].strip()

    if len(text) > 30:
        await message.answer(
            "❌ Watermark troppo lungo (max 30 caratteri)"
        )
        return

    if user_id not in sessions:
        sessions[user_id] = {}

    sessions[user_id]["custom_watermark"] = text

    await message.answer(
        f"✅ Watermark impostato:\n\n"
        f"<code>{text}</code>",
        parse_mode="HTML",
    )

@dp.callback_query(F.data.startswith("emoji:"))
async def emoji_callback(callback: types.CallbackQuery) -> None:
    user_id = callback.from_user.id
    session = sessions.get(user_id)
    
    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return
    
    emoji_code = callback.data.split(":", 1)[1]
    emoji_names = {
        "shocked": "😳 Classic",
        "peach": "🍑 Peach",
        "18": "🔞 NSFW",
        "lips": "💋 Lips",
        "black": "🖤 Black",
    }
    emoji_name = emoji_names.get(emoji_code, emoji_code)
    
    # Save emoji to session
    sessions[user_id]["emoji"] = emoji_code
    
    await callback.answer(f"✨ {emoji_name} selezionato")
    
    style = session.get("style")
    strength = session.get("strength", "medium")
    
    # Proceed to processing
    text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🛡️ <b>Vuoi aggiungere una firma?</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=session["status_msg_id"],
        text=text,
        reply_markup=WATERMARK_ENABLE_KEYBOARD,
        parse_mode="HTML",
    )


@dp.callback_query(F.data.startswith("wmpos:"))
async def watermark_position_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return

    position = callback.data.split(":", 1)[1]

    sessions[user_id]["watermark_position"] = position

    await callback.answer("✨ Posizione selezionata")

    text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔠 <b>Dimensione watermark</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=session["status_msg_id"],
        text=text,
        reply_markup=WATERMARK_SIZE_KEYBOARD,
        parse_mode="HTML",
    )

@dp.callback_query(F.data.startswith("wmsize:"))
async def watermark_size_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return

    size = callback.data.split(":", 1)[1]

    sessions[user_id]["watermark_size"] = size

    await callback.answer("✨ Dimensione selezionata")

    text = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "💧 <b>Opacità watermark</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=session["status_msg_id"],
        text=text,
        reply_markup=WATERMARK_OPACITY_KEYBOARD,
        parse_mode="HTML",
    )

@dp.callback_query(F.data.startswith("wmopacity:"))
async def watermark_opacity_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta", show_alert=True)
        return

    opacity = float(callback.data.split(":", 1)[1])

    sessions[user_id]["watermark_opacity"] = opacity

    await callback.answer("✨ Watermark configurato")

    style = session.get("style")
    strength = session.get("strength", "medium")

    await process_media_async(
        callback,
        user_id,
        session,
        style,
        strength,
    )

@dp.callback_query(F.data == "back")
async def back_callback(callback: types.CallbackQuery):

    user_id = callback.from_user.id
    session = sessions.get(user_id)

    if not session:
        await callback.answer("❌ Sessione scaduta")
        return

    style = session.get("style")
    strength = session.get("strength")
    target = session.get("target")

    status_msg_id = session["status_msg_id"]

    # BACK FROM TARGET → STYLE
    if not target and not strength:

        text = (
            "🎨 <b>Scegli la modalità di censura</b>"
        )

        await bot.edit_message_text(
            chat_id=user_id,
            message_id=status_msg_id,
            text=text,
            reply_markup=INLINE_KEYBOARD,
            parse_mode="HTML",
        )

        sessions[user_id].pop("strength", None)
        return

    # BACK FROM STRENGTH → TARGET
    if target and not strength:

        target_text = (
            "🛡️ <b>Cosa vuoi censurare?</b>"
        )

        await bot.edit_message_text(
            chat_id=user_id,
            message_id=status_msg_id,
            text=target_text,
            reply_markup=TARGET_KEYBOARD,
            parse_mode="HTML",
        )
        sessions[user_id].pop("target", None)
        return

    # BACK FROM WATERMARK/EMOJI → STRENGTH
    text = (
        "🎚️ <b>Seleziona intensità</b>"
    )

    await bot.edit_message_text(
        chat_id=user_id,
        message_id=status_msg_id,
        text=text,
        reply_markup=STRENGTH_KEYBOARD,
        parse_mode="HTML",
    )

async def fake_progress_bar(user_id: int, message_id: int):

    progress = 0

    while progress < 95:

        filled = progress // 10
        empty = 10 - filled

        bar = "▓" * filled + "░" * empty

        text = (
            "🎬 <b>Elaborazione video</b>\n\n"
            f"{bar} {progress}%\n\n"
            "🧠 AI censoring in corso..."
        )

        try:
            await bot.edit_message_text(
                chat_id=user_id,
                message_id=message_id,
                text=text,
                parse_mode="HTML",
            )
        except:
            pass

        await asyncio.sleep(2)

        progress += 5

async def process_media_async(callback: types.CallbackQuery, user_id: int, session: dict, style: str, strength: str) -> None:
    """Process media with selected style, strength, and emoji."""
    input_path = session["media_path"]
    media_type = session["media_type"]
    emoji = session.get("emoji", "shocked")
    target = session.get("target", "all")
    status_msg_id = session.get("status_msg_id")

    watermark_enabled = session.get(
        "watermark_enabled",
        True,
    )

    watermark_text = session.get("custom_watermark")

    watermark_position = session.get("watermark_position", "bottom")
    watermark_size = session.get("watermark_size", "medium")
    watermark_opacity = session.get("watermark_opacity", 0.5)

    if media_type == "video":
        progress_task = asyncio.create_task(
            fake_progress_bar(user_id, status_msg_id)
        )
        if video_processing_lock.locked():
            await callback.message.answer(
                "⏳ Un altro video è già in elaborazione.\n\n"
                "Riprova tra qualche istante."
            )
            return
    
    if media_type == "photo":
        output_path = os.path.join(session["temp_dir"], "output.jpg")
    else:
        output_path = os.path.join(session["temp_dir"], "output.mp4")
    
    try:
        # Show processing state
        processing_text = (
            f"⏳ <b>Elaborazione in corso...</b>\n\n"
            f"🎨 Stile: { {'blur': '🌫️ Blur', 'pixel': '🟪 Pixel', 'emoji': '😳 Emoji'}.get(style, style) }\n"
            f"🎚️  Intensità: {strength.capitalize()}\n\n"
            f"⚡ Potrebbe richiedere qualche secondo"
        )
        if status_msg_id:
            try:
                await bot.edit_message_text(
                    chat_id=user_id,
                    message_id=status_msg_id,
                    text=processing_text,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.debug("Cannot edit message: %s", e)
        
        # Process media
        if media_type == "photo":
            process_image(input_path,
                output_path,
                style,
                strength,
                emoji,
                target,
                watermark_text,
                watermark_position,
                watermark_size,
                watermark_opacity,
                watermark_enabled
            )
            file_obj = FSInputFile(output_path)
            await bot.send_photo(
                user_id,
                file_obj,
                caption="✅ <b>Contenuto censurato!</b>\n\n🛡️ <i>Processed by CoverIA</i>",
                parse_mode="HTML",
            )
        else:
            # Video processing with status updates
            fps, duration = get_video_metadata(input_path)
            
            # Update status for longer videos
            if duration > 30:
                detailed_text = (
                    f"⏳ <b>Elaborazione video in corso</b>\n\n"
                    f"🎞️  Estrazione frame...\n"
                    f"🤖 AI detection...\n"
                    f"🎬 Rendering finale...\n\n"
                    f"⚡ Questo potrebbe richiedere un minuto"
                )
                if status_msg_id:
                    try:
                        await bot.edit_message_text(
                            chat_id=user_id,
                            message_id=status_msg_id,
                            text=detailed_text,
                            parse_mode="HTML",
                        )
                    except Exception as e:
                        logger.debug("Cannot edit message: %s", e)
            async with video_processing_lock:
                process_video(
                    input_path,
                    output_path,
                    style,
                    strength,
                    emoji,
                    target,
                    watermark_text,
                    watermark_position,
                    watermark_size,
                    watermark_opacity,
                    watermark_enabled
                )

                if progress_task:
                    progress_task.cancel()

            file_obj = FSInputFile(output_path)
            await bot.send_video(
                user_id,
                file_obj,
                caption="✅ <b>Video censurato!</b>\n\n🛡️ <i>Processed by CoverIA</i>",
                parse_mode="HTML",
            )
        
        # Delete processing message if exists
        if status_msg_id:
            try:
                await bot.delete_message(chat_id=user_id, message_id=status_msg_id)
            except Exception as e:
                logger.debug("Cannot delete status message: %s", e)

    except RuntimeError as exc:
        error_text = (
            f"❌ <b>Errore durante l'elaborazione</b>\n\n"
            f"{str(exc)}\n\n"
            f"<b>Consigli:</b>\n"
            f"• Prova con un file più piccolo\n"
            f"• Video: usa formato <code>MP4</code>\n"
            f"• Durata massima: 2 minuti"
        )
        if status_msg_id:
            try:
                await bot.edit_message_text(
                    chat_id=user_id,
                    message_id=status_msg_id,
                    text=error_text,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.debug("Cannot edit message: %s", e)
                await callback.message.answer(error_text, parse_mode="HTML")
        else:
            await callback.message.answer(error_text, parse_mode="HTML")
        logger.error("Errore di elaborazione: %s", exc)
    except Exception as exc:
        error_text = (
            "❌ <b>Errore inaspettato</b>\n\n"
            "Contatta il supporto se il problema persiste."
        )
        if status_msg_id:
            try:
                await bot.edit_message_text(
                    chat_id=user_id,
                    message_id=status_msg_id,
                    text=error_text,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.debug("Cannot edit message: %s", e)
                await callback.message.answer(error_text, parse_mode="HTML")
        else:
            await callback.message.answer(error_text, parse_mode="HTML")
        logger.error("Errore di elaborazione: %s", exc)
    finally:
        cleanup_session(user_id)

async def main():
    logger.info("Avvio bot...")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrotto dall'utente")