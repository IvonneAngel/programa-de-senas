import { CSSProperties } from "react";

export interface FloatingEmoji {
  id: number;
  emoji: string;
  x: number; // percentage width
  scale: number;
  rotation: number;
  text?: string;
  splashPath?: string;
  splashGradient?: string;
  textStyle?: CSSProperties;
  imageUrl?: string;
}

export interface PlacedSticker {
  id: number;
  templateId: string;
  emoji: string;
  text: string;
  splashPath: string;
  splashGradient: string;
  textStyle: CSSProperties;
  x: number; // position percent
  y: number; // position percent
  scale: number;
  rotation: number;
  isMirrored: boolean;
  imageUrl?: string;
}

export interface StickerTemplate {
  id: string;
  emoji: string;
  text: string;
  splashPath: string;
  splashGradient: string;
  textStyle: CSSProperties;
  imageUrl?: string;
}

export type DatasetCategory = "letras_y_numeros" | "palabras";

export interface DatasetItem {
  id: string;
  name: string;
  category: DatasetCategory;
  subType: "letra" | "numero" | "palabra";
  frameCount?: number;
  custom?: boolean;
}

export interface CapturedFrame {
  id: string;
  categoryId: DatasetCategory;
  itemId: string;
  timestamp: number;
  dataUrl: string;
}

