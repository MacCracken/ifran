import { writable } from "svelte/store";
import type { Model } from "$lib/api/client";
import { listModels, deleteModel as apiDeleteModel } from "$lib/api/client";

export const models = writable<Model[]>([]);
export const loading = writable(false);
export const error = writable<string | null>(null);

export async function refreshModels() {
  loading.set(true);
  error.set(null);
  try {
    const result = await listModels();
    models.set(result.data);
  } catch (e) {
    error.set(String(e));
  } finally {
    loading.set(false);
  }
}

export async function removeModel(id: string) {
  await apiDeleteModel(id);
  await refreshModels();
}
