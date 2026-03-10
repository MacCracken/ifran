import { writable } from "svelte/store";
import type { SystemStatus, HardwareInfo } from "$lib/api/client";
import { getStatus, getHardware } from "$lib/api/client";

export const status = writable<SystemStatus | null>(null);
export const hardware = writable<HardwareInfo | null>(null);

export async function refreshStatus() {
  try {
    const s = await getStatus();
    status.set(s);
  } catch {
    status.set(null);
  }
}

export async function refreshHardware() {
  try {
    const h = await getHardware();
    hardware.set(h);
  } catch {
    hardware.set(null);
  }
}
