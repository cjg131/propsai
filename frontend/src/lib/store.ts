import { create } from "zustand";
import { persist } from "zustand/middleware";

interface BankrollState {
  bankroll: number;
  unitSize: number;
  setBankroll: (amount: number) => void;
  setUnitSize: (size: number) => void;
}

export const useBankrollStore = create<BankrollState>()(
  persist(
    (set) => ({
      bankroll: 1000,
      unitSize: 10,
      setBankroll: (amount) => set({ bankroll: amount }),
      setUnitSize: (size) => set({ unitSize: size }),
    }),
    { name: "propsai-bankroll" }
  )
);

interface SettingsState {
  preferredBooks: string[];
  activePreset: "conservative" | "balanced" | "aggressive" | "custom";
  fantasyFormat: "draftkings" | "fanduel" | "yahoo";
  setPreferredBooks: (books: string[]) => void;
  setActivePreset: (preset: SettingsState["activePreset"]) => void;
  setFantasyFormat: (format: SettingsState["fantasyFormat"]) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      preferredBooks: [],
      activePreset: "balanced",
      fantasyFormat: "draftkings",
      setPreferredBooks: (books) => set({ preferredBooks: books }),
      setActivePreset: (preset) => set({ activePreset: preset }),
      setFantasyFormat: (format) => set({ fantasyFormat: format }),
    }),
    { name: "propsai-settings" }
  )
);
