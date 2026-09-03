import { CSSProperties } from "react";
import { StickerTemplate } from "../types";

// Import beautiful generated sticker images
import imgQueOnda from "../assets/images/sticker_que_onda_1782931956231.jpg";
import imgTu from "../assets/images/sticker_tu_1782931972825.jpg";
import imgAlto from "../assets/images/sticker_alto_1782931982900.jpg";
import imgTeAmo from "../assets/images/sticker_te_amo_1782931992881.jpg";
import imgHola from "../assets/images/sticker_hola_1782932002669.jpg";
import imgBien from "../assets/images/sticker_bien_1782932013778.jpg";
import imgMal from "../assets/images/sticker_mal_1782932024305.jpg";
import imgGracias from "../assets/images/sticker_gracias_1782932035863.jpg";
import imgUno from "../assets/images/sticker_uno_1782932048100.jpg";
import imgDos from "../assets/images/sticker_dos_1782932058765.jpg";
import imgTres from "../assets/images/sticker_tres_1782932068590.jpg";
import imgOk from "../assets/images/sticker_ok_1782932077095.jpg";
import imgPoco from "../assets/images/sticker_poco_1782932089396.jpg";
import imgAdios from "../assets/images/sticker_adios_1782932098941.jpg";
import imgPienso from "../assets/images/sticker_pienso_1782932108389.jpg";
import imgTeAgradezco from "../assets/images/sticker_te_agradezco_1782932117858.jpg";

// Dynamic hand-drawn graffiti splash paths (scaled to fit nicely in viewBox)
const splashes = {
  purple: {
    gradient: "linear-gradient(135deg, #a855f7 0%, #6366f1 100%)",
    path: "M10,35 C25,48 55,50 75,38 C90,28 105,10 98,-15 C90,-35 70,-45 45,-48 C20,-50 -5,-35 -12,-15 C-20,5 -5,22 10,35 Z"
  },
  blue: {
    gradient: "linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)",
    path: "M15,25 C35,30 65,45 80,30 C95,15 100,-10 90,-30 C80,-50 50,-40 25,-45 C0,-50 -20,-30 -25,-10 C-30,10 -5,20 15,25 Z"
  },
  pink: {
    gradient: "linear-gradient(135deg, #ec4899 0%, #f43f5e 100%)",
    path: "M5,30 C20,45 50,55 70,40 C85,28 105,5 95,-18 C85,-38 65,-42 40,-46 C15,-50 -10,-35 -18,-18 C-25,0 -10,15 5,30 Z"
  },
  green: {
    gradient: "linear-gradient(135deg, #10b981 0%, #047857 100%)",
    path: "M12,28 C28,32 58,48 78,35 C92,22 98,-5 88,-25 C78,-45 48,-38 22,-42 C-4,-46 -24,-28 -28,-8 C-32,12 -8,24 12,28 Z"
  },
  orange: {
    gradient: "linear-gradient(135deg, #f59e0b 0%, #ea580c 100%)",
    path: "M8,32 C22,46 52,52 72,38 C88,25 102,2 92,-20 C82,-40 62,-44 38,-48 C14,-52 -12,-38 -19,-20 C-26,-2 -10,18 8,32 Z"
  }
};

// Common high-contrast graffiti outline style for text to ensure absolute legibility
const createTextStyle = (textColor: string = "#ffffff"): CSSProperties => ({
  textShadow: "2.5px 2.5px 0px #000, -2.5px -2.5px 0px #000, 2.5px -2.5px 0px #000, -2.5px 2.5px 0px #000, 4px 4px 0px rgba(0,0,0,0.45)",
  color: textColor,
});

export const ALL_TEMPLATES: StickerTemplate[] = [
  // --- ROW 1 ---
  {
    id: "que_onda",
    emoji: "👊",
    text: "¿QUÉ ONDA?",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#fef08a"), // Yellow text
    imageUrl: imgQueOnda
  },
  {
    id: "tu",
    emoji: "👉",
    text: "TÚ",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgTu
  },
  {
    id: "alto",
    emoji: "✋",
    text: "ALTO",
    splashPath: splashes.pink.path,
    splashGradient: splashes.pink.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgAlto
  },
  {
    id: "te_amo",
    emoji: "🤟",
    text: "TE AMO",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#a7f3d0"), // Light green text
    imageUrl: imgTeAmo
  },
  {
    id: "hola",
    emoji: "👋",
    text: "HOLA",
    splashPath: splashes.green.path,
    splashGradient: splashes.green.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgHola
  },
  {
    id: "uno",
    emoji: "☝️",
    text: "UNO",
    splashPath: splashes.orange.path,
    splashGradient: splashes.orange.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgUno
  },

  // --- ROW 2 ---
  {
    id: "dos",
    emoji: "✌️",
    text: "DOS",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgDos
  },
  {
    id: "tres",
    emoji: "🤟",
    text: "TRES",
    splashPath: splashes.orange.path,
    splashGradient: splashes.orange.gradient,
    textStyle: createTextStyle("#fef08a"),
    imageUrl: imgTres
  },
  {
    id: "cuatro",
    emoji: "🖐️",
    text: "CUATRO",
    splashPath: splashes.pink.path,
    splashGradient: splashes.pink.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "cinco",
    emoji: "🖐️",
    text: "CINCO",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "bien",
    emoji: "👍",
    text: "BIEN",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#f472b6"), // Pink text
    imageUrl: imgBien
  },
  {
    id: "mal",
    emoji: "👎",
    text: "MAL",
    splashPath: splashes.green.path,
    splashGradient: splashes.green.gradient,
    textStyle: createTextStyle("#fca5a5"), // Reddish text
    imageUrl: imgMal
  },

  // --- ROW 3 ---
  {
    id: "ok",
    emoji: "👌",
    text: "OK",
    splashPath: splashes.pink.path,
    splashGradient: splashes.pink.gradient,
    textStyle: createTextStyle("#fffbeb"),
    imageUrl: imgOk
  },
  {
    id: "poco",
    emoji: "🤏",
    text: "POCO",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgPoco
  },
  {
    id: "mucho",
    emoji: "🫴",
    text: "MUCHO",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "tu_pointing",
    emoji: "🫵",
    text: "TÚ",
    splashPath: splashes.orange.path,
    splashGradient: splashes.orange.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgTu // Reuse imgTu
  },
  {
    id: "yo",
    emoji: "👈",
    text: "YO",
    splashPath: splashes.green.path,
    splashGradient: splashes.green.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "gracias",
    emoji: "🙏",
    text: "GRACIAS",
    splashPath: splashes.pink.path,
    splashGradient: splashes.pink.gradient,
    textStyle: createTextStyle("#fed7aa"), // Light orange text
    imageUrl: imgGracias
  },

  // --- ROW 4 ---
  {
    id: "por_favor",
    emoji: "🤲",
    text: "POR FAVOR",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "perdon",
    emoji: "🙏",
    text: "PERDÓN",
    splashPath: splashes.green.path,
    splashGradient: splashes.green.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgGracias // Reuse folded hands
  },
  {
    id: "ven_aca",
    emoji: "🫵",
    text: "VEN ACÁ",
    splashPath: splashes.orange.path,
    splashGradient: splashes.orange.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "adios",
    emoji: "👋",
    text: "ADIÓS",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgAdios
  },
  {
    id: "escucho",
    emoji: "👂",
    text: "ESCUCHO",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "pienso",
    emoji: "🤔",
    text: "PIENSO",
    splashPath: splashes.pink.path,
    splashGradient: splashes.pink.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgPienso
  },

  // --- ROW 5 ---
  {
    id: "gusto_conocerte",
    emoji: "🤝",
    text: "GUSTO EN CONOCERTE",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "te_agradezco",
    emoji: "🫶",
    text: "TE AGRADEZCO",
    splashPath: splashes.pink.path,
    splashGradient: splashes.pink.gradient,
    textStyle: createTextStyle("#ffffff"),
    imageUrl: imgTeAgradezco
  },
  {
    id: "comer",
    emoji: "🤌",
    text: "COMER",
    splashPath: splashes.purple.path,
    splashGradient: splashes.purple.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "tomar",
    emoji: "☕",
    text: "TOMAR",
    splashPath: splashes.blue.path,
    splashGradient: splashes.blue.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "escribir",
    emoji: "✍️",
    text: "ESCRIBIR",
    splashPath: splashes.green.path,
    splashGradient: splashes.green.gradient,
    textStyle: createTextStyle("#ffffff")
  },
  {
    id: "leer",
    emoji: "📖",
    text: "LEER",
    splashPath: splashes.orange.path,
    splashGradient: splashes.orange.gradient,
    textStyle: createTextStyle("#ffffff")
  }
];

export const STICKER_TEMPLATES: StickerTemplate[] = ALL_TEMPLATES.filter(
  (t) => t.imageUrl !== undefined
);
